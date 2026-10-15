# fast_ppo: faster MATLAB PPO training (2026-10-16)

Separate method (`fast_ppo` in `external_ref/ldpc_rl/train/sched_train.m`, `matlab_fast_ppo`
in the database); `ppo` is unchanged. Same network, loss and update (`lib/ppo_update.m`); a
trained fast_ppo policy is an ordinary `'ppo'` policy and is evaluated by the same decoder.

| Change | Where | Effect on results |
|---|---|---|
| Batched rollouts: up to `batch_frames` (64) frames advance one step together | `lib/fast_ppo_rollout.m` | none: `lib/fast_ppo_smoketest.m` checks it against `lib/sched_frame.m` frame by frame (greedy: identical outcomes, bit-identical LLRs; training: identical actions/rewards/lengths for the same draws, log-prob/value diffs <= 4e-15, MATLAB and Octave) |
| Parallel collection (`collect_workers`), per-(round, worker) Threefry substreams | `lib/fast_ppo_collect.m` | implemented, but the profile showed no gain once rollouts are batched (it slows the update); runs use 1 worker |
| 2 update epochs (ppo: 4) | config `ppo.update_epochs` | hyperparameter change, part of the config |
| Train at 1.5-3 dB (1 dB dropped: ~all frames fail, no success-reward signal) | `sched_driver.py::FAST_TRAIN_SNRS` | protocol change, part of the config |
| Compute threads for the update | `sched_driver.py::TRAIN_THREADS` | execution only |

Training stops after the first collection round reaching `episodes` (overshoot <= one round,
~70 episodes at z=4); `meta.episodes` records the actual count. The evaluation shard
(`parallel_eval/sched_eval_shard.m`) now decodes PPO-kind policies in batches of 250 with the
same noise draws as before; counts are identical to the per-frame path.

Measured on Ada (MATLAB R2024a, ms per training episode): z=4 ppo 117, fast_ppo 57 (1 thread),
37 (4 threads); z=16 12.4 -> 6.4; z=1 ~1900 (ppo 3k run) -> 276 (8 threads, 10k run).
