"""
Build expdb config dicts for MATLAB-produced results.

The one rule this module enforces: every method name originating from MATLAB
carries the ``matlab_`` prefix. MATLAB results are produced by
``external_ref/ldpc_rl/`` — a different codebase, a different training rule,
and a different scheduling loop from this repo's Python ``rl/``. Mixing them
into the experiment database under an unprefixed name would make two genuinely
different algorithms look like one method with noisy results.

The prefix is enforced here rather than left to the caller so it cannot be
forgotten at a call site.
"""
from __future__ import annotations

MATLAB_PREFIX = "matlab_"

# Keys copied verbatim from a MATLAB run manifest into the expdb config, if
# present. Deliberately mirrors run_experiments.py's config shape so MATLAB
# and Python rows for the same matrix line up when compared.
#
# REQUIRED: present for every MATLAB method this bridge ingests, learned or
# not. scheduling is required (not merely optional-if-present) because it is
# what distinguishes a genuine reproduction of Python's rl/trainer.py
# mechanics (full sweep per iteration, own-cluster Q bootstrap) from the
# original MATLAB reference mechanics (raw greedy picks, global-max
# bootstrap) -- see docs/notes/matlab_bridge_quartile_k3.md and the
# 2026-09-15 correction in docs/notes/archived_results/. Silently defaulting
# it would let two algorithmically different runs collide under the same
# config_id.
_REQUIRED_KEYS = (
    "matrix",
    "z",
    "seed",
    "state_encoding",
    "scheduling",
)

# OPTIONAL, TRAINING: only meaningful for a learned method (Q-learning). A
# deterministic baseline like round-robin has no training phase at all --
# forcing alpha/gamma/train_episodes onto it would misrepresent it as a
# trained model that happens to have those hyperparameters, so these are
# omitted from its config entirely rather than filled with placeholder
# values. (This deliberately avoids the inconsistency already documented in
# CLAUDE.md where run_experiments.py forces these fields onto every Python
# method including non-learned ones.)
_OPTIONAL_TRAINING_KEYS = (
    "alpha",
    "gamma",
    "epsilon",
    "l_max",
    "train_episodes",
    "train_snr_vals",
)

# OPTIONAL, ARCHITECTURE-SPECIFIC: included only when the manifest provides
# them, since different architectures parameterize their state encoding
# differently (quartile's "k" vs. a hypothetical binary encoder's
# "max_state_bits", etc.). Each such field IS part of model identity for the
# architecture that has it -- it's just not universal.
_OPTIONAL_ARCH_KEYS = (
    "k",
    "max_state_bits",
    "eval_maxIter",
)

_OPTIONAL_KEYS = _OPTIONAL_TRAINING_KEYS + _OPTIONAL_ARCH_KEYS

_CONFIG_KEYS = _REQUIRED_KEYS + _OPTIONAL_KEYS


class MethodPrefixError(ValueError):
    """Raised when a MATLAB result carries a method name without the prefix."""


def enforce_matlab_prefix(method: str) -> str:
    """
    Return ``method`` guaranteed to start with ``matlab_``.

    Raises rather than silently rewriting when the name looks like it is
    already trying to claim a Python method name (e.g. plain ``reldec``),
    because silently renaming would hide a real mix-up between the two
    codebases. An empty/missing method is always an error — there is no
    safe default to fall back to.
    """
    if not method or not isinstance(method, str):
        raise MethodPrefixError(f"method must be a non-empty string, got {method!r}")
    if method.startswith(MATLAB_PREFIX):
        return method
    raise MethodPrefixError(
        f"MATLAB-produced result has method={method!r}, which lacks the "
        f"required {MATLAB_PREFIX!r} prefix. MATLAB results must be "
        f"distinguishable from this repo's Python rl/ runs in the experiment database. "
        f"Fix the producing script (run_quartile_eval.m sets this) rather "
        f"than renaming here."
    )


def config_from_manifest(manifest: dict) -> dict:
    """
    Build the expdb config dict from a MATLAB run manifest.

    Only the model-identity fields are included; eval-budget fields
    (target_frame_errors, max_frames) are intentionally left out of the config
    because expdb hashes configs with those in HASH_EXCLUSIONS anyway — they
    live in the eval_results primary key instead.
    """
    method = enforce_matlab_prefix(manifest.get("method", ""))

    missing = [k for k in _REQUIRED_KEYS if k not in manifest]
    if missing:
        raise KeyError(
            f"manifest is missing required config keys: {missing}. "
            f"Refusing to write a partial config to the experiment database."
        )

    cfg = {k: manifest[k] for k in _CONFIG_KEYS if k in manifest}
    cfg["method"] = method

    # Normalise MATLAB's scalar-vs-array quirks: jsonencode emits scalars for
    # 1-element arrays, and everything numeric comes back as float. Only
    # applies when train_snr_vals is present -- deterministic baselines
    # (round-robin, etc.) have no training phase and omit it entirely.
    if "train_snr_vals" in cfg and not isinstance(cfg["train_snr_vals"], list):
        cfg["train_snr_vals"] = [cfg["train_snr_vals"]]
    for int_key in ("z", "k", "l_max", "train_episodes", "seed", "eval_maxIter", "max_state_bits"):
        if int_key in cfg and cfg[int_key] is not None:
            cfg[int_key] = int(cfg[int_key])

    return cfg
