# Ada job scripts for the MATLAB cluster-scheduling runs (Oct 2026)

These are the exact SLURM scripts that produced the `matlab_*` rows in the experiment
database. They run from a working copy at `~/Decoder_100k` on Ada and call
`matlab_bridge/sched_driver.py`, which defines every stage (code, method, z, episodes,
SNRs, seeds) and skips stages already complete in the database.
See `docs/notes/matlab_sched_runs.md` for the run record.

| Script | What it ran |
|---|---|
| `install_matlab.sh` | Installs MATLAB R2024a + Parallel Computing Toolbox into `~/matlab_R2024a` with `mpm` (licensed by the user's MathWorks online login; Ada's network license server failed with -97) |
| `job_octave_setup.sh` | Pulls the GNU Octave 9.2 container (`docker://gnuoctave/octave:9.2.0`) used while MATLAB was unavailable |
| `job_train_oct.sh`, `job_eval_oct.sh` | Octave versions of training/evaluation (produced WRAN flooding and the four WRAN 500-episode tabular policies) |
| `convert_ckpt.m` | Re-saved those Octave checkpoints in MATLAB `-v7` format (values unchanged) |
| `job_train.sh`, `job_eval.sh` | Main MATLAB run: all `STAGES` (WRAN z=1/4, BG2 Z=10) |
| `job_eval36.sh` | Re-ran the two BG2 15k tabular stages that failed (parpool worker crash) and finished BG2 PPO 3k on 36 workers |
| `job_ppo10k.sh` | BG2 PPO 10k multithreaded training + retry (training was cancelled; no 10k results) |
| `job_tanh_sweep.sh` | tanh-MI WRAN z=1 episode sweep (2 and 3 dB) |
| `job_ep_sweep2.sh` | tanh-MI + RELDEC WRAN z=4,16 episode sweep (2 and 3 dB) |
| `job_ep1k_full.sh` | 1k-episode tanh-MI + RELDEC WRAN z=1,4,16 at 1-6 dB |
| `launch.sh` | Laptop-side rsync + submit helper |

Trained policies are in `external_ref/ldpc_rl/checkpoints/sched/<stage>.mat`.
