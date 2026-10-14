"""
Driver for the MATLAB cluster-scheduling runs (external_ref/ldpc_rl/train/sched_train.m
and parallel_eval/run_sched_eval.m) on Ada.

Every stage is defined here once. For each stage the driver builds the exact
manifest the ingester will hash into a config_id, and SKIPS the stage when the
database already holds max_frames frames at every SNR for that config, so a
rerun (or a second submission) never repeats finished work.

    python -m matlab_bridge.sched_driver train-list OUT.json
        write the sched_train configs still needed (checkpoint missing and the
        stage's evaluation not already complete)
    python -m matlab_bridge.sched_driver train-list OUT_DIR --per-config
        same, one JSON file per config (for launching each training separately)
    python -m matlab_bridge.sched_driver eval --workers N --train-done FILE [--engine octave]
        evaluate stages in order of availability: baselines at once, learned
        stages as soon as their checkpoint exists. Fails loudly if training
        finished (FILE exists) without producing a checkpoint.
    python -m matlab_bridge.sched_driver status
        print each stage and its database coverage
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expdb import get_coverage
from expdb.config import compute_config_hash
from matlab_bridge.config import config_from_manifest

ROOT = Path(__file__).resolve().parent.parent
MLAB = ROOT / "external_ref" / "ldpc_rl"
CKPT = MLAB / "checkpoints" / "sched"
SPOOL = ROOT / "spool" / "sched"
MAX_FRAMES = 100_000
CHUNK = 1000

CODES = {
    # code: (repo matrix path, i_max, eval SNRs, train SNRs)
    "wran": ("matrices/WRAN_irreg_384_256.csv", 5, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 1.5, 2.0, 2.5, 3.0]),
    "bg2_z10": ("matrices/H_BG2_Z10.csv", 50, [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5],
                [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
}
TABULAR = {"alpha": 0.1, "gamma": 0.99, "epsilon": 0.1}       # same as run_experiments.py
PPO_BASE = {"lr": 1e-3, "gamma": 1.0, "gae_lambda": 0.95, "clip_eps": 0.2, "update_epochs": 4,
            "minibatch_size": 512, "rollout_steps": 8192, "ent_coef": 0.003, "vf_coef": 0.5,
            "max_grad_norm": 0.5, "hidden": 32, "reward": "success", "pos_feature": False}
PPO_BG2 = dict(PPO_BASE, gamma=0.999, pos_feature=True, rollout_steps=32768, minibatch_size=2048)

METHOD = {"flooding": "matlab_flooding", "round_robin": "matlab_round_robin", "reldec": "matlab_reldec",
          "tanh_mi": "matlab_ave_tanh_ave_mi", "ppo": "matlab_global_ppo"}
STATE = {"flooding": "none", "round_robin": "none", "reldec": "binary_vector",
         "tanh_mi": "ave_tanh_21bin", "ppo": "ppo_cluster_features"}
SCHED = {"flooding": "flooding", "round_robin": "round_robin_fixed_order", "reldec": "sweep",
         "tanh_mi": "sweep", "ppo": "sweep"}

# (code, kind, z, train_episodes or None, eval base seed)
# WRAN round_robin z=1/z=4 is deliberately absent: matlab_round_robin already has
# 100k frames at every SNR for both (docs/notes/matlab_bridge_round_robin.md).
STAGES = [
    ("wran", "flooding", 1, None, 41001),
    ("wran", "reldec", 1, 500, 41011), ("wran", "reldec", 4, 500, 41014),
    ("wran", "tanh_mi", 1, 500, 41021), ("wran", "tanh_mi", 4, 500, 41024),
    ("wran", "ppo", 1, 3000, 41031), ("wran", "ppo", 4, 3000, 41034),
    ("bg2_z10", "flooding", 1, None, 42001), ("bg2_z10", "round_robin", 1, None, 42002),
    ("bg2_z10", "reldec", 1, 15000, 42011), ("bg2_z10", "tanh_mi", 1, 15000, 42021),
    ("bg2_z10", "ppo", 1, 3000, 42031),
    # "trained for more episodes" variants
    ("bg2_z10", "reldec", 1, 50000, 42012), ("bg2_z10", "tanh_mi", 1, 50000, 42022),
    ("bg2_z10", "ppo", 1, 10000, 42032),
]


# Episode sweep for tanh-MI on WRAN z=1 (2026-10-14): only 2 and 3 dB are evaluated.
# The 500-episode point is the same config as STAGES' wran_tanh_mi_z1_ep500 (reused).
EP_SWEEP = [("wran", "tanh_mi", 1, ep, 43000 + i) for i, ep in enumerate([10, 100, 500, 1000, 3000, 5000])]
EVAL_SNRS = {f"wran_tanh_mi_z1_ep{ep}": [2.0, 3.0] for ep in (10, 100, 1000, 3000, 5000)}
# Episode sweep for tanh-MI and RELDEC on WRAN z=4 and z=16 (2026-10-14), 2 and 3 dB only.
# The z=4 500-episode points are STAGES' wran_{reldec,tanh_mi}_z4_ep500 (reused).
_EPS = [10, 100, 500, 1000, 3000, 5000]
EP_SWEEP_Z4_Z16 = [("wran", kind, z, ep, 44000 + 100 * z + 10 * j + i)
                   for z in (4, 16) for j, kind in enumerate(("tanh_mi", "reldec")) for i, ep in enumerate(_EPS)]
for _st in EP_SWEEP_Z4_Z16:
    if not (_st[2] == 4 and _st[3] == 500):
        EVAL_SNRS[f"wran_{_st[1]}_z{_st[2]}_ep{_st[3]}"] = [2.0, 3.0]
# 1k-episode tanh-MI and RELDEC on WRAN z = 1, 4, 16 at the standard 1..6 dB (2026-10-14).
# Same configs (and checkpoints) as the episode sweeps' 1k points, which already hold 2 and 3 dB;
# a new eval seed per stage keeps the added SNR points' noise independent of those.
EP1K_FULL = [("wran", kind, z, 1000, 45000 + 10 * z + j, CODES["wran"][2])
             for z in (1, 4, 16) for j, kind in enumerate(("tanh_mi", "reldec"))]
STAGE_SETS = {"main": STAGES, "tanh_ep_sweep": EP_SWEEP, "ep_sweep_z4_z16": EP_SWEEP_Z4_Z16, "ep1k_full": EP1K_FULL}


def eval_snrs(st):
    """A 6th tuple element, if present, is the stage's explicit eval SNR list."""
    if len(st) > 5:
        return st[5]
    return EVAL_SNRS.get(stage_name(st), CODES[st[0]][2])


def stage_name(st) -> str:
    code, kind, z, ep, _ = st[:5]
    return f"{code}_{kind}_z{z}" + (f"_ep{ep}" if ep else "")


def ppo_hyper(code):
    return PPO_BG2 if code == "bg2_z10" else PPO_BASE


def manifest(st) -> dict:
    code, kind, z, ep, eval_seed = st[:5]
    matrix, i_max, _, train_snrs = CODES[code]
    m = {"method": METHOD[kind], "matrix": matrix, "z": z, "state_encoding": STATE[kind],
         "scheduling": SCHED[kind], "eval_maxIter": i_max,
         "target_frame_errors": MAX_FRAMES, "max_frames": MAX_FRAMES,
         "matrix_matlab": "external_ref/ldpc_rl/matrices/" + ("wran_384_256.mat" if code == "wran" else "P_520.mat (block 10)")}
    if ep is None:
        m["seed"] = eval_seed                     # baselines: seed = eval seed (matlab_round_robin convention)
    else:
        m.update(seed=42, l_max=i_max, train_episodes=ep, train_snr_vals=train_snrs)
        if kind == "ppo":
            m["ppo"] = ppo_hyper(code)
        else:
            m.update(TABULAR)
    return m


def train_cfg(st) -> dict:
    code, kind, z, ep, _ = st[:5]
    _, i_max, _, train_snrs = CODES[code]
    c = {"code": code, "z": z, "method": kind, "episodes": ep, "l_max": i_max, "train_snrs": train_snrs,
         "seed": 42, "out": str(CKPT / f"{stage_name(st)}.mat"), "threads": 1}
    if kind == "ppo":
        c["ppo"] = ppo_hyper(code)
    else:
        c.update(TABULAR)
    return c


def selected(skip, stage_set="main"):
    """The chosen stage set minus those whose name contains any of the --skip substrings."""
    return [st for st in STAGE_SETS[stage_set] if not any(x in stage_name(st) for x in (skip or []))]


def config_id(st) -> str:
    return compute_config_hash(config_from_manifest(manifest(st)))


def complete(st) -> bool:
    cov = get_coverage(config_id(st), MAX_FRAMES, MAX_FRAMES)
    return all(cov.get(s, {}).get("frames_done", 0) >= MAX_FRAMES for s in eval_snrs(st))


def n_shards(st) -> int:
    return len(eval_snrs(st)) * -(-MAX_FRAMES // CHUNK)


def cmd_status(args):
    for st in STAGE_SETS[args.stage_set]:
        cov = get_coverage(config_id(st), MAX_FRAMES, MAX_FRAMES)
        frames = [cov.get(s, {}).get("frames_done", 0) for s in eval_snrs(st)]
        print(f"{stage_name(st):32s} cfg={config_id(st)[:8]} frames/SNR={frames}")


def cmd_train_list(args):
    need = [st for st in selected(args.skip, args.stage_set) if st[3] and not complete(st) and not Path(train_cfg(st)["out"]).exists()]
    # longest first, so the parfor starts the slow trainings immediately
    weight = {"ppo": 50, "reldec": 1, "tanh_mi": 1}
    need.sort(key=lambda st: -(st[3] * weight[st[1]] * (100 if st[0] == "bg2_z10" else 1)))
    CKPT.mkdir(parents=True, exist_ok=True)
    if args.per_config:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for st in need:
            (out / f"{stage_name(st)}.json").write_text(json.dumps(train_cfg(st)))
    else:
        Path(args.out).write_text(json.dumps([train_cfg(st) for st in need], indent=1))
    print(f"{len(need)} training configs -> {args.out}: " + ", ".join(stage_name(st) for st in need))


def run_stage(st, workers, py, engine="matlab"):
    name = stage_name(st)
    spool = SPOOL / name
    spool.mkdir(parents=True, exist_ok=True)
    man = dict(manifest(st), n_shards=n_shards(st), chunk_size=CHUNK, engine=engine,
               engine_detail=os.environ.get("OCTAVE_CMD", "matlab") if engine == "octave" else "matlab")
    (spool / "manifest.json").write_text(json.dumps(man))
    code, kind, z, ep, eval_seed = st[:5]
    cfg = {"manifest": manifest(st), "code": code, "z": z, "kind": kind,
           "snr_db": eval_snrs(st), "n_frames": MAX_FRAMES, "chunk_size": CHUNK,
           "i_max": CODES[code][1], "base_seed": eval_seed, "n_workers": workers, "spool_dir": str(spool)}
    if ep:
        cfg["checkpoint"] = train_cfg(st)["out"]
    cfg_path = spool / "eval_cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    print(f"[{time.strftime('%H:%M:%S')}] stage {name} cfg={config_id(st)[:8]}", flush=True)
    ing = subprocess.Popen([py, "-u", "-m", "matlab_bridge.ingest", "--spool-dir", str(spool),
                            "--expected-chunks", str(n_shards(st)), "--poll-interval", "5"], cwd=ROOT)
    if engine == "matlab":
        ml = subprocess.run(os.environ.get("MATLAB_CMD", "matlab").split() + ["-batch",
                             f"addpath('{MLAB / 'parallel_eval'}'); run_sched_eval('{cfg_path}')"], cwd=MLAB)
        rc = ml.returncode
    else:
        # Octave's parfor is serial: run `workers` independent processes, each on
        # every workers-th shard of the same sweep. OCTAVE_CMD is the command
        # prefix (e.g. "singularity exec --bind $HOME image.sif octave-cli").
        oct = os.environ["OCTAVE_CMD"].split()
        procs = []
        for part in range(1, workers + 1):
            pc = dict(cfg, part=part, nparts=workers)
            pp = spool / f"eval_cfg_part{part}.json"
            pp.write_text(json.dumps(pc))
            log = open(spool / f"octave_part{part}.log", "w")
            procs.append(subprocess.Popen(
                oct + ["--no-gui", "--quiet", "--eval",
                       f"addpath('{MLAB / 'parallel_eval'}'); run_sched_eval('{pp}')"],
                cwd=MLAB, stdout=log, stderr=subprocess.STDOUT))
        rc = max(p.wait() for p in procs)
    if rc != 0:
        ing.terminate()
        raise RuntimeError(f"{engine} failed for stage {name} (exit {rc})")
    try:
        ing.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        ing.terminate()
        raise RuntimeError(f"ingester for {name} did not finish within 30 min of MATLAB exiting")
    if ing.returncode != 0 or not complete(st):
        raise RuntimeError(f"stage {name}: ingester exit {ing.returncode}, database coverage incomplete")
    print(f"[{time.strftime('%H:%M:%S')}] stage {name} complete", flush=True)


def cmd_eval(args):
    stages = selected(args.skip, args.stage_set)
    pending = [st for st in stages if not complete(st)]
    for st in stages:
        if st not in pending:
            print(f"skip {stage_name(st)}: already complete in the database", flush=True)
    failed = []
    while pending:
        ready = [st for st in pending if st[3] is None or Path(train_cfg(st)["out"]).exists()]
        if not ready:
            if Path(args.train_done).exists():
                for st in pending:
                    print(f"ERROR {stage_name(st)}: training finished but {train_cfg(st)['out']} is missing",
                          file=sys.stderr, flush=True)
                failed += pending
                break
            time.sleep(60)
            continue
        st = ready[0]
        pending.remove(st)
        try:
            run_stage(st, args.workers, sys.executable, args.engine)
        except Exception as e:                       # keep going with the other stages, report at the end
            print(f"ERROR {stage_name(st)}: {e}", file=sys.stderr, flush=True)
            failed.append(st)
    if failed:
        print("FAILED stages: " + ", ".join(stage_name(st) for st in failed), file=sys.stderr)
        return 1
    print("ALL STAGES COMPLETE", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="matlab_bridge.sched_driver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st_ = sub.add_parser("status"); st_.add_argument("--stage-set", choices=list(STAGE_SETS), default="main")
    t = sub.add_parser("train-list"); t.add_argument("out")
    t.add_argument("--per-config", action="store_true")
    t.add_argument("--stage-set", choices=list(STAGE_SETS), default="main")
    t.add_argument("--skip", nargs="*", default=[], help="skip stages whose name contains any of these")
    e = sub.add_parser("eval"); e.add_argument("--workers", type=int, required=True)
    e.add_argument("--train-done", required=True)
    e.add_argument("--engine", choices=["matlab", "octave"], default="matlab")
    e.add_argument("--stage-set", choices=list(STAGE_SETS), default="main")
    e.add_argument("--skip", nargs="*", default=[], help="skip stages whose name contains any of these")
    args = ap.parse_args(argv)
    return {"status": cmd_status, "train-list": cmd_train_list, "eval": cmd_eval}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
