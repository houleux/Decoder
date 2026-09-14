# Quartile-average-LLR RELDEC architecture (MATLAB, `external_ref/`)

Date: 2026-09-11

## What

Added a new RELDEC state-encoding architecture, entirely under `external_ref/ldpc_rl/`
(the third-party MATLAB reference tree), per explicit user request:

- **State** for each cluster (one cluster = one check node, same convention as
  `train/reldec_test.m`): sort the LLRs of the check node's neighboring
  variable nodes ascending, split them into `k` groups as evenly as possible,
  and take the average LLR within each group. `k = 4` (quartiles) is the
  default, but `k` is a parameter — the same trainer runs for `k = 3, 4, 5`.
- **Reward**: unchanged in *form* from `train/reldec_test.m` — the fraction of
  the cluster's state slots that are "satisfied" (hard-decoded `>= 0`) after
  the update, with a correction term to cancel the artificial credit that
  zero-padded slots would otherwise contribute. `k` substitutes for
  `maxStateBits` in that formula.
- Matrix: `wran_384_256.mat` (128×384, row degree 10–11, rate 2/3), the same
  matrix `reldec_test.m` itself defaults to.
- Trained and evaluated for `k ∈ {3, 4, 5}`, 1000 frames per SNR point, SNR_db
  = [0 1 2 3 4], BER/FER reported and plotted.

New files (all under `external_ref/ldpc_rl/`, which is excluded from the "no
new files without permission" rule here because the user explicitly asked for
this architecture to be added there):

| File | Role |
|---|---|
| `lib/quartile_state.m` | Reference (unoptimized, one check node at a time) state encoder |
| `lib/build_deg_groups.m` | Groups check nodes by degree for vectorized batch computation |
| `lib/quartile_state_batch.m` | Vectorized state encoder for all `m` check nodes at once — verified numerically identical to `quartile_state.m` (see below) |
| `lib/RELDEC_QUARTILE_MAIN.m` | Tabular Q-learning trainer (the actual training loop) |
| `train/reldec_quartile_llr.m` | Training driver: builds the graph, generates training LLRs, trains `k = 3, 4, 5`, saves one checkpoint per `k` |
| `eval/test_reldec_quartile_llr.m` | Eval driver: greedy-policy BP scheduling, BER/FER over the SNR sweep, summary table, `semilogy` comparison plot |

Checkpoints land in `checkpoints/Q_wran_quartile_k{3,4,5}.mat`; eval results in
`checkpoints/quartile_llr_eval_results.mat` and `checkpoints/quartile_llr_ber.png`.

## Why

User request: add this state architecture, run it for `k = 3, 4, 5`, report
1000-frame results with a plot, on the WRAN matrix, reusing `reldec_test.m`'s
reward. This is a research comparison against the existing reference
architectures already in `external_ref/`, not a change to this repo's Python
`rl/`/`global_mdp/` code — nothing under `rl/` or `global_mdp/` was touched.

## Design decisions

- **State/reward mapping.** `reldec_test.m`'s reward formula operates on a
  fixed-size hard-decision bit vector derived from raw neighbor LLRs
  (`maxStateBits` slots, zero-padded if a check node's degree is smaller).
  The quartile architecture's state is also a fixed-size vector (`k` slots),
  just built from *quantile-group averages* instead of raw per-neighbor LLRs.
  Reusing the reward verbatim meant substituting `k` for `maxStateBits`
  everywhere that formula references padding/degree, so the "same reward"
  claim in the request maps onto a state of a different, smaller size rather
  than literally reusing `maxStateBits` (which was tied to per-neighbor
  slots, and the assignment doesn't apply to a k-length quantile vector).
- **Padding correction is defensive, not load-bearing.** WRAN's check-node
  degree is 10–11, always ≥ k for k ∈ {3,4,5}, so the zero-padding path
  (`k_eff < k`) never triggers in this run. It exists so the same trainer file
  stays correct if pointed at a lower-degree matrix later.
- **R (code rate) bug fixed, not inherited.** `reldec_test.m` (the file this
  reward is copied from) references an undefined variable `R` — a real bug in
  the third-party reference (documented in `CLAUDE.md`'s per-script table).
  The new trainer computes `R = 1 - m/n` directly from `H` instead of leaving
  it undefined or requiring the Communications Toolbox's `ldpcEncoderConfig`
  just to read off a rate that's derivable from the matrix itself.
- **Split the training loop into its own function file
  (`lib/RELDEC_QUARTILE_MAIN.m`) instead of a script-trailing local function.**
  All the upstream reference scripts (`reldec_test.m`, `rl_nips_test.m`, etc.)
  define their training loop as a MATLAB script with a `function ... end`
  block appended after the script body — valid MATLAB, but confirmed **not**
  supported by GNU Octave 11.1 (the interpreter actually available in this
  environment; MATLAB itself is not installed here). A minimal repro
  (`x=...; y=foo(x); function y=foo(x) ... end`) fails in Octave with `'foo'
  undefined`. Using a separate function file is portable to both and is
  otherwise a purely mechanical difference from upstream's style.
- **Vectorized (`quartile_state_batch.m`) instead of the reference scripts'
  per-check-node loop.** The reference architectures all recompute state for
  every check node with an explicit `for i = 1:m` loop each step — fine in
  MATLAB's JIT, intractably slow in Octave's plain interpreter at the scale
  this task needs (3 k-values × 5 SNR points × 1000 frames × up to 640
  scheduling steps/frame). `build_deg_groups.m` groups check nodes by degree
  (2 groups for WRAN: degree 10 and 11) so the sort+average happens as one
  vectorized matrix op per degree instead of `m=128` separate calls.
  **Verified numerically identical** to the reference per-node
  `quartile_state.m` (`max abs diff = 0` across k = 3, 4, 5 on a random LLR
  vector) before being wired into the trainer/eval loops. `quartile_state.m`
  itself is kept as the readable, unoptimized reference definition of the
  state encoder and is what the reward computation still calls.
- **No Communications Toolbox in the new eval script.** `eval/test_reldec.m`
  (the existing reference eval harness) uses `ldpcEncode`/`ldpcEncoderConfig`
  to build a proper codeword. Since RELDEC's own reward/training scripts
  never actually encode a message (they generate all-zero-codeword BPSK+AWGN
  LLRs directly across all `n` bits, e.g. `rx = 1 + sigma*randn(1,n)`), the
  new eval script does the same and reports BER/FER over all `n` bits, not
  just systematic/information bits — there's no encoder config here to
  isolate them, and using the toolbox for that alone seemed like an
  unjustified new dependency for this comparison. This does mean the BER
  numbers aren't directly comparable to a proper information-bit BER from a
  toolbox-based decoder; flagged in the results note.
- **Octave, not MATLAB, actually ran this.** MATLAB is not installed in this
  environment; `apt-get install octave` was run (with the user's explicit
  go-ahead) to get a working interpreter for the actual training/eval run
  described in `docs/notes/quartile_llr_reldec_results.md`. The scripts
  are plain MATLAB (`.m`) and should run unmodified in real MATLAB with the
  Communications Toolbox not required for this particular pair.
- **Eval scheduling budget:** `maxIter = 5` "sweeps" (`m*maxIter` = 640
  scheduling actions per frame with early stop the moment the syndrome is
  all-zero), matching the `maxnumiter = 5` convention already used by
  `eval/test_reldec.m`.

See `docs/notes/quartile_llr_reldec_results.md` for the actual numbers, the
run configuration used, and any caveats found while running it.
