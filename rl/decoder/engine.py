import csv
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy.sparse as sp
from rl.channel import awgn_llr


@dataclass
class MethodStats:
    method: str
    n: int
    frames: int = 0
    bit_errors: int = 0
    frame_errors: int = 0
    converged_frames: int = 0
    messages: int = 0
    iterations: int = 0

    def update(self, tx_bits: np.ndarray, result) -> None:
        self.frames += 1
        errs = int(np.count_nonzero(tx_bits != result.bits))
        self.bit_errors += errs
        if errs > 0:
            self.frame_errors += 1
        if result.converged:
            self.converged_frames += 1
        self.iterations += result.iterations
        self.messages += result.messages

    @property
    def ber(self) -> float:
        return self.bit_errors / max(1, self.frames * self.n)

    @property
    def fer(self) -> float:
        return self.frame_errors / max(1, self.frames)


def _worker(args: tuple) -> MethodStats:
    """
    Top-level worker function (must be picklable — no lambdas, no closures).

    args is a tuple of:
        h_data, h_indices, h_indptr, h_shape,  ← raw CSR components
        method: str,
        z: int,
        checkpoint_path: str | None,
        ebn0_db: float,
        code_rate: float,
        i_max: int,
        n_frames: int,
        target_frame_errors: int,
        seed: int,
    """
    (
        h_data, h_indices, h_indptr, h_shape,
        method, z, checkpoint_path,
        ebn0_db, code_rate, i_max,
        n_frames, target_frame_errors, seed,
    ) = args

    h_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    # Build decoder inside worker (BpDecoder is not picklable)
    if method == "flooding":
        from rl.decoder.flooding import FloodingDecoder
        decoder = FloodingDecoder(h_csr)
    elif method == "round_robin":
        from rl.decoder.sequential import RoundRobinDecoder
        decoder = RoundRobinDecoder(h_csr)
    elif method == "random":
        from rl.decoder.sequential import RandomDecoder
        decoder = RandomDecoder(h_csr, rng)
    elif method == "rbl":
        from rl.decoder.rbl import ResidualDecoder
        decoder = ResidualDecoder(h_csr)
    elif method == "ave_rbl":
        from rl.decoder.ave_rbl import AveRBLDecoder
        decoder = AveRBLDecoder(h_csr, z)
    elif method == "max_rbl":
        from rl.decoder.max_rbl import MaxRBLDecoder
        decoder = MaxRBLDecoder(h_csr, z)
    elif method == "reldec":
        from rl.agents.reldec import ReldecAgent
        agent = ReldecAgent.load(checkpoint_path, h_csr)
        decoder = agent  # ReldecAgent implements decode()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    stats = MethodStats(method=method, n=h_shape[1])
    tx_bits = np.zeros(h_shape[1], dtype=np.uint8)  # All-zero codeword only

    while stats.frames < n_frames and stats.frame_errors < target_frame_errors:
        llr = awgn_llr(h_shape[1], ebn0_db, code_rate, rng)
        result = decoder.decode(llr, i_max)
        stats.update(tx_bits, result)

    return stats


def _merge(stats_list: list[MethodStats]) -> MethodStats:
    merged = MethodStats(method=stats_list[0].method, n=stats_list[0].n)
    for s in stats_list:
        merged.frames += s.frames
        merged.bit_errors += s.bit_errors
        merged.frame_errors += s.frame_errors
        merged.converged_frames += s.converged_frames
        merged.messages += s.messages
        merged.iterations += s.iterations
    return merged


def evaluate_snr_point(
    h_csr: sp.csr_matrix,
    method: str,
    z: int,
    checkpoint_path: str | None,
    ebn0_db: float,
    code_rate: float,
    i_max: int,
    target_frame_errors: int,
    max_frames: int,
    rng: np.random.Generator,
    n_workers: int = 1,
) -> MethodStats:
    """
    Evaluate a single SNR point by running up to max_frames frames.
    Stops early when target_frame_errors is reached.
    """
    frames_each = max(1, max_frames // n_workers)
    frames_last = max_frames - frames_each * (n_workers - 1)
    errors_each = max(1, target_frame_errors // n_workers)
    errors_last = target_frame_errors - errors_each * (n_workers - 1)
    base_seed = int(rng.integers(0, 2**31))

    h = h_csr.tocsr().astype(np.uint8)
    h_args = (h.data, h.indices, h.indptr, h.shape)

    worker_args = [
        (
            *h_args,
            method, z, checkpoint_path,
            ebn0_db, code_rate, i_max,
            (frames_each if i < n_workers - 1 else frames_last),
            (errors_each if i < n_workers - 1 else errors_last),
            base_seed + i,
        )
        for i in range(n_workers)
    ]

    if n_workers == 1:
        return _worker(worker_args[0])

    partial = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed([pool.submit(_worker, a) for a in worker_args]):
            partial.append(fut.result())
    return _merge(partial)


def write_csv(results: list[tuple[float, MethodStats]], output_path: str) -> None:
    """
    Write evaluation results to a CSV file.

    Args:
        results:     List of (ebn0_db, MethodStats) pairs, one per SNR point.
        output_path: Path to write the CSV file.

    Columns: method, ebn0_db, frames, bit_errors, frame_errors, ber, fer, avg_iterations, avg_messages, converged_frames
    """
    fieldnames = ["method", "ebn0_db", "frames", "bit_errors", "frame_errors",
                  "ber", "fer", "avg_iterations", "avg_messages", "converged_frames"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ebn0_db, s in results:
            writer.writerow({
                "method": s.method,
                "ebn0_db": ebn0_db,
                "frames": s.frames,
                "bit_errors": s.bit_errors,
                "frame_errors": s.frame_errors,
                "ber": s.ber,
                "fer": s.fer,
                "avg_iterations": s.iterations / max(1, s.frames),
                "avg_messages": s.messages / max(1, s.frames),
                "converged_frames": s.converged_frames,
            })
