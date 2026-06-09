# Agent Constitution

## Design Values

> Guiding principles for all code changes and additions.

1. **Understand before acting.** Read `docs/architecture.md` before making any changes. Map your planned change to existing structures — prefer extending what exists over inventing something new.
2. **Minimize footprint.** Do not create new files without explicit user permission, except for experiment configuration files. When in doubt, ask.
3. **Write for scale.** If new files are approved, make them modular, structured, and reusable. Every new abstraction should earn its place.
4. **No silent fallbacks.** It is unacceptable to introduce default behaviors, silent fallbacks, or broad exception handling that could silently alter the results and mislead the user (researcher). Always fail loudly and explicitly require configuration instead of guessing intent.

---

## Documentation Protocol

> How to track, communicate, and consolidate changes over time.

### Per-Change Notes

- For every meaningful change (a new feature, a significant bug fix, a refactor), create **one markdown file** in `docs/notes/`.
- Name files descriptively: e.g., `docs/notes/add-beam-search-decoder.md`, `docs/notes/fix-llr-overflow.md`.
- Each note should briefly describe: **what** changed, **why**, and **any design decisions** made.

### Architecture Sync

- At regular intervals, request user approval to **purge notes into `docs/architecture.md`** and delete the individual note files.
- Keep `docs/architecture.md` as the single source of truth for the codebase structure.

### Constraints

- Do **not** create any other markdown files anywhere in the codebase unless explicitly instructed.
- The only sanctioned markdown locations are:
  - `docs/architecture.md` — living architecture reference
  - `docs/notes/` — transient per-change notes (pending purge)
