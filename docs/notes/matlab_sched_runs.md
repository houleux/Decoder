# MATLAB cluster-scheduling runs (2026-10-08 to 10-14)

New MATLAB implementation of the cluster-scheduling decoders in `external_ref/ldpc_rl/`,
written to follow this repo's Python semantics (not the original `reldec_test.m` raw-pick
mechanics), and the runs made with it. Full replication spec (equations, seeds, files):
the PDF `matlab_replication_spec` produced alongside this note (LaTeX source in
`docs/notes/matlab_replication_spec.tex`).

## Code
- `lib/sched_env.m` graph/clusters (WRAN `wran_384_256.mat`; BG2 Z=10 from `P_520.mat`, block 10)
- `lib/sched_frame.m` one frame: flooding-free sequential BP, round-robin / RELDEC / tanh-MI / PPO
  policies, training updates. `lib/flood_frame.m` flooding. `lib/mi_llr.m` J-function MI.
- `lib/ppo_*.m` PPO network, hand-written gradient (`ppo_gradcheck.m`), Adam update.
- `lib/sched_policy_pack.m` / `unpack.m`: RELDEC hash-map state store (neighborhoods > 52 bits, z=16).
- `train/sched_train.m` (+ `_batch`, `_json`), `parallel_eval/run_sched_eval.m` + `sched_eval_shard.m`.
- `matlab_bridge/sched_driver.py` stage definitions, DB-skip, ingestion; `matlab_bridge/ada_jobs/` job scripts.

## Verification
Flooding and layered decoders frame-for-frame identical to Python (iterations, convergence,
messages); tabular FER within noise of Python; PPO update matches torch to ~5e-6 and PPO
features exactly (`tests` not added; checks were run ad hoc, see session record).

## Runs (all 100k frames/SNR, results in the database under `matlab_*` methods)
- WRAN z=1,4 (l_max=5, 1-6 dB): flooding, RELDEC, tanh-MI (500 ep), PPO (3k ep).
- BG2 Z=10 (l_max=50, 1.75-3.5 dB): flooding, round-robin, RELDEC/tanh-MI (15k and 50k ep), PPO (3k ep).
  PPO 10k cancelled.
- Episode sweeps (2 and 3 dB): tanh-MI z=1; tanh-MI and RELDEC z=4,16 (10..5000 ep).
- 1k-episode tanh-MI/RELDEC z=1,4,16 at 1-6 dB.

## Deviations worth knowing
- Ada's MATLAB license server returned -97; MATLAB was installed per-user and licensed with the
  user's online login. WRAN flooding and the four WRAN 500-ep tabular trainings ran in Octave 9.2
  (same .m code; Octave seeds with randn/rand('state') instead of Threefry).
- Two BG2 15k tabular stages failed once (MATLAB parpool worker crash at startup) and were re-run.
- MATLAB WRAN PPO z=1 is worse than Python PPO (FER 0.416 vs 0.221 at 2 dB); implementations were
  verified equivalent, so this is single-seed training variance.
