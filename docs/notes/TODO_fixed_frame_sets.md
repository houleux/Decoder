# TODO (on hold, 2026-10-15): fixed training / evaluation frame sets for the MATLAB runs

Status: agreed design, not implemented. Parked at the user's request in favour of the
PPO episode sweep.

## Why
Every MATLAB evaluation so far drew its own frames (each stage has its own `base_seed`,
each SNR its own substreams), so method-vs-method comparisons are unpaired. Large
differences are unaffected; close calls (e.g. tanh-MI vs RELDEC at z=4, neighbouring
episode counts, high-SNR points with a handful of errors) cannot be resolved. Python
`run_experiments.py` runs were paired only when run with the same worker count.

## Design
- **Eval set E** (100,000 noise vectors, not stored): frame f = 1..100000 uses
  `RandStream('Threefry','Seed',S_eval)` with `Substream = f`, `w_f = randn(st,1,n)`;
  at any SNR the frame is `L = 2(1 + sigma*w_f)/sigma^2`. Same frames for every method,
  z and SNR; x < 100k frames means frames 1..x; independent of chunking and worker count.
- **Train set T**: same construction with seed S_train; episode e uses frame e (at the
  cycled training SNR). Never exceed 100k episodes.
- **Tie-breaks**: per-frame substream of a third seed so tabular evaluation is
  bit-reproducible.
- **Optional held-out test set** (seed S_test) if hyperparameters/episode counts are chosen
  by looking at E.
- Add `eval_set` / `train_set` labels to the config identity so results on E/T never mix
  with the existing database rows.
- MATLAB-only (Octave has no RandStream; Python can't reproduce MATLAB Threefry normals);
  storing E/T as files (~150-400 MB per code, float32) would make them portable.

## Integrity notes
Disjoint seeds => no train/eval overlap; training on T with <100k episodes is equivalent
to fresh frames; SNR points along a curve become correlated (each still unbiased);
reporting on the same set used for tuning is optimistic -> use the test set for final numbers.

## Estimated cost (WRAN z=1,4,16; RELDEC + tanh-MI 1k episodes, PPO 3k; 100k frames x 1-6 dB)
Implementation ~1 h; training ~40 min (critical path PPO z=1, estimate), overlapped;
evaluation ~2 h 10 min on 35 workers (PPO z=1 alone ~65 min); ~2.5 h on Ada, ~3.5 h end to end.
A test set doubles the evaluation. Add flooding/round-robin on E (~5 min) for paired baselines.
