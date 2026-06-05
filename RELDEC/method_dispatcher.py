from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from RELDEC.algorithms.reldec_deep import (
    load_deep_decoder_from_checkpoint,
    MiReldecBaselineDecoder,
    MiTabularQDecoder,
    evaluate_mi_tabular_method,
)
from RELDEC.algorithms.reldec_tabular_augmented import TabularAugmentedQDecoder
from RELDEC.algorithms.reldec_augmented import load_augmented_deep_decoder_from_checkpoint
from RELDEC.algorithms.reldec_core import load_q_table, load_parity_check_from_sparse_csv


class MethodDispatcher:
    """Instantiate decoders for each method based on its family and requirements."""

    def __init__(
        self,
        matrix_csv: str | Path,
        h_csr: sp.csr_matrix | None = None,
        q_table_path: str | Path | None = None,
        mi_tabular_q_table_path: str | Path | None = None,
        tabular_augmented_q_table_path: str | Path | None = None,
        deep_checkpoint_path: str | Path | None = None,
        mi_bins: int = 21,
        args_z: int | None = None,
    ):
        self.matrix_csv = Path(matrix_csv)
        # avoid evaluating a scipy.sparse matrix's truth value (raises ValueError)
        self.h_csr = h_csr if h_csr is not None else load_parity_check_from_sparse_csv(self.matrix_csv)
        self.q_table = load_q_table(q_table_path) if q_table_path else None
        self.mi_tabular_q_table = load_q_table(mi_tabular_q_table_path) if mi_tabular_q_table_path else None
        self.tabular_augmented_q_table = load_q_table(tabular_augmented_q_table_path) if tabular_augmented_q_table_path else None
        self.deep_checkpoint_path = deep_checkpoint_path
        self.mi_bins = int(mi_bins)
        self.args_z = args_z
        self._deep_decoders_cache: Dict[str, Any] = {}
        self._mi_naive_cache: Dict[str, Any] = {}

    def _load_deep_decoder(self, method: str, policy_label: str) -> Any:
        if method not in self._deep_decoders_cache:
            if method.startswith("augmented_"):
                decoder = load_augmented_deep_decoder_from_checkpoint(
                    checkpoint_path=str(self.deep_checkpoint_path),
                    matrix_csv=str(self.matrix_csv),
                    expected_policy_label=policy_label,
                )
            else:
                decoder = load_deep_decoder_from_checkpoint(
                    checkpoint_path=str(self.deep_checkpoint_path),
                    matrix_csv=str(self.matrix_csv),
                    expected_policy_label=policy_label,
                )
            self._deep_decoders_cache[method] = decoder
        return self._deep_decoders_cache[method]

    def get_decoder(self, method: str) -> Any:
        """Get or create a decoder for the given method."""
        if method == "reldec":
            from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
            suite = ReldecDecoderSuite(self.h_csr)
            if self.q_table is not None:
                suite.set_q_table(self.q_table)
            return suite

        elif method == "deep_reldec_z1":
            return self._load_deep_decoder(method, "deep_z1")
        elif method == "deep_reldec_zx":
            return self._load_deep_decoder(method, "deep_zx")

        elif method == "mi_dqn_zx":
            return self._load_deep_decoder(method, "mi_dqn_zx")

        elif method == "augmented_max_avg_zx":
            return self._load_deep_decoder(method, "augmented_max_avg_zx")
        elif method == "augmented_max_zx":
            return self._load_deep_decoder(method, "augmented_max_zx")
        elif method == "augmented_average_zx":
            return self._load_deep_decoder(method, "augmented_average_zx")

        elif method == "mi_naive_zx":
            if "mi_naive_zx" not in self._mi_naive_cache:
                if self.args_z is None:
                    raise ValueError("args_z required for mi_naive_zx")
                self._mi_naive_cache["mi_naive_zx"] = MiReldecBaselineDecoder(self.h_csr, cluster_size=int(self.args_z))
            return self._mi_naive_cache["mi_naive_zx"]

        elif method == "mi_tabular_zx":
            if self.mi_tabular_q_table is None:
                raise ValueError("mi_tabular_q_table required for mi_tabular_zx")
            if self.args_z is None:
                raise ValueError("args_z required for mi_tabular_zx")
            return MiTabularQDecoder(
                h_csr=self.h_csr,
                q_table=self.mi_tabular_q_table,
                cluster_size=int(self.args_z),
                mi_bins=self.mi_bins,
            )

        elif method.startswith("tabular_augmented_"):
            if self.tabular_augmented_q_table is None:
                raise ValueError(f"tabular_augmented_q_table required for {method}")
            if self.args_z is None:
                raise ValueError(f"args_z required for {method}")
            return TabularAugmentedQDecoder(
                h_csr=self.h_csr,
                q_table=self.tabular_augmented_q_table,
                policy_label=method,
                cluster_size=int(self.args_z),
                mi_bins=self.mi_bins,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_decoders_for_methods(self, methods: list[str]) -> Dict[str, Any]:
        """Pre-allocate decoders for multiple methods."""
        decoders = {}
        for method in methods:
            if method not in {"flooding", "random", "round_robin"}:
                decoders[method] = self.get_decoder(method)
        return decoders
