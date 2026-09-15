# Bug: `external_ref/ldpc_rl/lib/` was never actually committed

Date discovered: 2026-09-15 (found by another agent running on a different
machine against the same GitHub repo — its checkout was missing the entire
`external_ref/ldpc_rl/lib/` directory and couldn't run
`train/reldec_quartile_llr.m` / `eval/test_reldec_quartile_llr.m`, both of
which `addpath('../lib')`).

## What was wrong

Root `.gitignore` (line 21, from the stock toptal Python template) had:

```
lib/
```

unanchored. Git matches an unanchored pattern against a directory of that
name **anywhere** in the tree, not just at repo root. The template's intent
was to ignore a top-level Python packaging build directory (`./lib/` from a
`setup.py bdist`-style build), but as written it also silently swallowed
`external_ref/ldpc_rl/lib/` — a real, hand-written MATLAB directory created
during the `external_ref/` reorg (`docs/notes/external_ref_reorganization.md`)
and later populated with the quartile-LLR architecture's shared functions
(`docs/notes/quartile_llr_reldec.md`).

Consequence: **every file ever placed in that directory** — not just the 4
new quartile files, all 9 pre-existing ones moved there during the reorg too
(`Jfun.m`, `Jinv.m`, `ldpc_cluster.m`, `ldpc_layered.m`,
`ldpc_residue_cluster.m`, `ldpcdec_cluster.m`, `ldpcdec_edge.m`,
`remove_one_row_submatrices.m`, `scheduler_c_v.m`) — was untracked. `git
add -A` / whole-directory commits never picked them up because `git status`
and `git add` both silently skip ignored paths by default. Commit `c9c5624`
("added quartile llr") shows deletions from the old flat layout but no
corresponding `lib/*.m` additions — the move was committed, the destination
never was. This was never caught locally because the files still physically
existed on the machine that did the reorg/training — `git status` was clean,
nothing looked wrong until a *different* machine cloned the repo fresh and
the directory was simply missing.

## Fix

Anchored the whole "Distribution / packaging" block in `.gitignore` to repo
root (`lib/` → `/lib/`, same for `build/`, `dist/`, `var/`, etc. — all were
equally unanchored and equally capable of silently swallowing a
similarly-named directory elsewhere in the tree; none currently do, checked
directly with `find . -type d -name <pattern>` across the whole repo before
anchoring, so this only *prevents* future recurrences, it didn't uncover
more hidden damage). `.venv/` is separately ignored on its own line, so
anchoring `lib/` does not expose `.venv/lib`.

Then `git add`ed all 13 files that were sitting untracked in
`external_ref/ldpc_rl/lib/` on this machine and committed them for real.
Nothing was reconstructed from scratch — the files were present here the
whole time, just never in git.

## Lesson

An unanchored `.gitignore` pattern copied from a stock template is a
directory-name landmine for the whole repo, not just the framework it was
written for. `git status`/`git add -A` being clean is not proof a directory
is committed — check `git ls-files <path>` (or `git check-ignore -v <path>`)
directly when something depends on a directory surviving a fresh clone,
especially right after creating a new directory that didn't exist when the
`.gitignore` was written.
