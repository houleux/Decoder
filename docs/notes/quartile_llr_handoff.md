# Handoff: quartile-average-LLR RELDEC (MATLAB, `external_ref/`)

Date: 2026-09-14. Status: **done and run once, end to end; not re-run with
parallelism yet.** This file is the "pick this back up" summary; the design
rationale and the actual results live in the two files linked below and are
not repeated here in full.

- Design/why: [`quartile_llr_reldec.md`](quartile_llr_reldec.md)
- Results + comparison vs. Python `rl/reldec`: [`quartile_llr_reldec_results.md`](quartile_llr_reldec_results.md)

## One-paragraph summary

Added a new RELDEC state-encoding architecture to the third-party MATLAB
reference tree at `external_ref/ldpc_rl/` (not this repo's Python `rl/` or
`global_mdp/` — those are untouched). State for a cluster (one check node) is
a vector of size `k`: sort the check node's neighbor LLRs, split into `k`
groups, average each. Trained/evaluated for `k = 3, 4, 5` on
`wran_384_256.mat`, reward = `reldec_test.m`'s formula with `k` substituted
for `maxStateBits`. Ran via Octave 11.1 (installed for this — MATLAB is not
available in this environment). Result: **k doesn't matter (3 vs 4 vs 5 are
statistically indistinguishable)**, and the architecture is **substantially
weaker than this repo's own Python hard-decision `rl/reldec`** on the same
matrix/clustering/decode-budget (see results doc for the ~5-orders-of-
magnitude BER gap by SNR=4dB).

## Where everything is

| What | Path |
|---|---|
| State encoder (reference, unvectorized) | `external_ref/ldpc_rl/lib/quartile_state.m` |
| State encoder (vectorized, verified identical) | `external_ref/ldpc_rl/lib/build_deg_groups.m` + `external_ref/ldpc_rl/lib/quartile_state_batch.m` |
| Training loop | `external_ref/ldpc_rl/lib/RELDEC_QUARTILE_MAIN.m` |
| Training driver (k=3,4,5) | `external_ref/ldpc_rl/train/reldec_quartile_llr.m` |
| Eval driver (BER/FER sweep + plot) | `external_ref/ldpc_rl/eval/test_reldec_quartile_llr.m` |
| Trained checkpoints | `external_ref/ldpc_rl/checkpoints/Q_wran_quartile_k{3,4,5}.mat` |
| Eval results (numeric) | `external_ref/ldpc_rl/checkpoints/quartile_llr_eval_results.mat` |
| Eval plot | `external_ref/ldpc_rl/checkpoints/quartile_llr_ber.png` |
| Raw run logs | `scratch/quartile_llr_run/{train,eval,pipeline}.log` (gitignored — `scratch/` never gets committed, so these don't survive past this machine; the numbers they back are already transcribed into `quartile_llr_reldec_results.md`) |
| Repo-level pointer | `CLAUDE.md`, "Added architecture: quartile-average-LLR RELDEC" section under `## external_ref/` |
| Committed | `git log`: `c9c5624 added quartile llr` (and `c8acd35 added bhagav matlab code` — the `external_ref/` reorg that preceded it) |

## Environment this ran in

- **MATLAB is not installed here.** `apt-get install -y octave` was run
  (12 vCPU / 7.6 GiB container, confirmed via `nproc`/`free -h`) to get GNU
  Octave 11.1.0 as a stand-in interpreter. The `.m` files are plain MATLAB
  and should run unmodified in real MATLAB; the only place the two
  interpreters actually diverged was **script-trailing local functions**
  (`function ... end` appended after script code) — every upstream reference
  script uses that form, Octave 11.1 does not support it, so
  `RELDEC_QUARTILE_MAIN.m` was split into its own function file instead.
  Confirmed with a minimal repro before assuming this, not guessed.
- No Octave packages are installed (`pkg list` → none) — in particular, no
  `parallel` package, which matters for the next-steps section below.

## What's *not* done — the concrete next step

The user asked (2026-09-14) whether the MATLAB code would run faster on
multi-CPU or multi-GPU systems. Short version of that analysis (full
reasoning is in the conversation, not yet copied into a docs/notes file —
worth doing if this becomes a recurring question):

- **Multi-GPU: no benefit without a rewrite.** Per-step work is a `tanh`/
  `atanh` over 10–11 elements; the graph is cell arrays + a sparse matrix
  (poor/no `gpuArray` support); action selection is scalar, data-dependent
  control flow. This is close to a worst-case GPU workload — kernel-launch
  and transfer overhead would dominate actual compute. Real GPU benefit
  would require batching many frames/episodes into dense tensors, i.e. a
  different algorithm.
- **Multi-CPU: yes, and we left it on the table.** Three different
  granularities, three different verdicts:
  1. **Eval, across frames — embarrassingly parallel, ran serially.** Every
     frame in `eval/test_reldec_quartile_llr.m` is independent (reads a fixed
     trained Q-table, no shared mutable state). This is exactly why the
     *original* reference eval scripts (`eval/test_reldec.m` etc.) use
     `parfor` (Parallel Computing Toolbox). Our new eval script used a plain
     serial loop for Octave-portability reasons and took **~2 hours**
     (3 k × 5 SNR × 1000 frames, up to 640 actions/frame). With 12 cores,
     this is very plausibly a **10–15 min job** if parallelized.
  2. **Training, across the 3 `k` values — trivially parallel, ran serially.**
     `k=3/4/5` are fully independent trainings (separate Q-tables). Ran
     back-to-back in one process (~58 min total: 16m/19m/23m for k=3/4/5).
     Running as 3 concurrent processes would cut this to roughly the slowest
     one (~23 min).
  3. **Training, within one `k`'s 30000 episodes — NOT a free parallelization.**
     Each episode reads *and writes* the same shared Q-table via in-place
     `alpha`-blended updates, and **update order matters** (this is online
     Q-learning). Naively parallelizing episodes races writes to the shared
     Q-table and silently changes what's learned. Doing this "properly" means
     Hogwild!-style async Q-learning or episode-batched updates — a real
     algorithmic decision, not an engineering freebie, and not something to
     do without checking with the user first (changes the learned policy,
     not just wall-clock time).

**If picking this up:** the highest-value, lowest-risk next step is
parallelizing (1) — the eval sweep — since it changes nothing about the
learned Q-tables or the reported numbers, only wall-clock time. Options,
roughly in order of least-to-most invasive:
- Shard the eval driver by `(k, SNR_db)` into up to 15 independent Octave
  processes (one per cell of the results table), run concurrently via shell
  job control or `xargs -P`, merge the resulting `.mat`/log outputs. No code
  change to the numerical logic — same script, different slice of the sweep
  per process.
- Or install Octave's `parallel` package (`pkg install -forge parallel`,
  needs network access to Octave Forge) and use `parcellfun`/`pararrayfun` to
  parallelize the frame loop directly.
- In real MATLAB, the straightforward fix is just wrapping the `for f =
  1:numFrames` loop in `parfor` (Parallel Computing Toolbox), matching what
  the original reference eval scripts already do.

None of this was implemented — the user asked the performance question but
did not (yet) ask for the re-run. Confirm before spending another ~hour of
wall-clock re-running the sweep, parallelized or not — the current numbers in
`quartile_llr_reldec_results.md` are real and already usable; a parallel
re-run would only produce the same numbers faster, not new information.

## Open items / things a future session should know

- The results doc flags a real training-procedure confound versus the Python
  `rl/reldec` comparison (forced one-action-per-cluster-per-sweep vs. our
  free repeated ε-greedy picks) — not resolved, would need a controlled
  ablation to isolate "is it the state encoding or the training procedure"
  causing the performance gap.
- BER is reported over all `n=384` bits (no encoder config available to
  isolate information bits) — confirmed this matches the Python `rl/` eval
  convention too (`rl/decoder/engine.py`: `bit_errors/(frames*n)`), so it's
  not a comparability gap, just worth remembering if comparing against a
  third source that *does* isolate information bits.
- FER was 1.0 in every one of the 15 (k, SNR) cells at these SNRs/budget —
  the architecture never reaches a clean-decode regime in this sweep. Not
  investigated further (e.g. whether a larger action budget, more training,
  or a different SNR range would change this).
