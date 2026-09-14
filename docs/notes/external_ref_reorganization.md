# `external_ref/` reorganization and documentation

Date: 2026-09-11

## What

Two changes, both confined to `external_ref/` and `CLAUDE.md`:

1. **Reorganized `external_ref/ldpc_rl/`** from a flat 113-file directory into
   `train/`, `eval/`, `lib/`, `matrices/`, `checkpoints/`, `misc/`, plus a new
   `external_ref/archive/` holding `superseded/` and `matlab_autosave/`.
2. **Replaced the short `external_ref/` section in `CLAUDE.md`** with a full one
   documenting the new layout, the three key trainers
   (`reldec_test.m`, `reldec_residue.m`, `rl_nips_test.m`), their state/reward/Q
   shapes, their inputs and `.mat` outputs, required MATLAB toolboxes, and what
   was archived.

No MATLAB source was edited. No Python in this repo was touched.

## Why

The directory was an undifferentiated dump — trainers, eval harnesses, library
functions, parity-check matrices, and ~50 trained Q-tables all at one level, with
no way to tell which script produced which `Q_*.mat`. The three scripts the user
actually compares against (`reldec_test.m`, `reldec_residue.m`, `rl_nips_test.m`)
were indistinguishable from their own superseded drafts.

## Design decisions

- **Split by role, not by code variant.** Trainer/eval/library/data is the axis
  that makes the dependency direction (`eval/` → `lib/` + `checkpoints/`,
  `train/` → `matrices/` → `checkpoints/`) legible. Splitting by RL variant
  (`reldec/`, `nips/`, `tanh_mi/`) would have scattered the shared `lib/`
  functions and duplicated matrix fixtures.
- **Archive rather than delete.** Per the user's instruction, each of three
  near-identical pairs was resolved to one keeper and the loser moved to
  `external_ref/archive/superseded/`: `reldec.m` (older, `P_520`-hardcoded,
  uncorrected reward) lost to `reldec_test.m`; `rl_tanh_mi.m` (passes a raw LLR
  to `Jfun`, which expects a noise sigma) lost to `rl_tanh_mi_fixed.m`;
  `ldpc_cluster_residue.m` (indexes a cell array as a matrix — cannot run) lost
  to `ldpc_residue_cluster.m`. The 9 `.asv` MATLAB editor autosaves went to
  `archive/matlab_autosave/` without asking, as they are backups, not sources.
- **Kept `eval/test_reldec.m` and `eval/test_reldec_residue_reward.m` both**, at
  the user's direction: identical decode loop, but distinct experiments
  (BlockSize 4 vs 1, different checkpoint sets, 100k vs 20k frames).
- **Did not fix the broken `load` paths.** Every script uses bare filenames and
  assumed a flat directory, so the move breaks them until MATLAB is started in
  `external_ref/ldpc_rl` with `addpath(genpath(pwd))`. Rewriting ~30 third-party
  scripts to add path prefixes would have made future diffs against upstream
  useless; the requirement is documented in `CLAUDE.md` instead.

## Correction to prior documentation

`CLAUDE.md` previously asserted `external_ref/` is gitignored. It is not — it is
tracked, and `git check-ignore` confirms no rule matches it. This reorganization
therefore appears in `git status` as ~113 deletions plus new untracked paths
(git will resolve them as renames on commit). The false claim is corrected in the
rewritten section.
