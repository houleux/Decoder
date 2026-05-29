"""Evaluation router - maps methods to their evaluation functions."""

from __future__ import annotations

from typing import Any
import numpy as np

from RELDEC.algorithms.reldec_core import ReldecDecoderSuite, evaluate_single_method
from RELDEC.algorithms.reldec_deep import evaluate_deep_method, evaluate_mi_tabular_method, MiReldecBaselineDecoder
from RELDEC.algorithms.reldec_deep import MiTabularQDecoder


def evaluate_method_with_dispatcher(
    dispatcher,
    method: str,
    snr_db: float,
    code_rate: float,
    i_max: int,
    target_frame_errors: int,
    max_frames: int,
    rng: np.random.Generator,
    all_zero_only: bool,
    suite: ReldecDecoderSuite | None = None,
) -> Any:
    """Evaluate a method using the appropriate evaluation function.
    
    Args:
        dispatcher: MethodDispatcher instance
        method: Method name (e.g., "reldec", "deep_reldec_z1", "mi_naive_z2")
        suite: Optional ReldecDecoderSuite for "reldec" and baseline methods
        Other args: Standard evaluation parameters
        
    Returns:
        DecodingStats object from the evaluation function
    """
    
    if method in {"flooding", "random", "round_robin", "reldec"}:
        # These use the suite and evaluate_single_method
        if suite is None:
            raise ValueError(f"suite is required for method {method}")
        return evaluate_single_method(
            suite=suite,
            method=method,
            snr_db=float(snr_db),
            code_rate=float(code_rate),
            i_max=i_max,
            target_frame_errors=int(target_frame_errors),
            max_frames=int(max_frames),
            rng=rng,
            all_zero_only=bool(all_zero_only),
        )
    
    elif method in {"deep_reldec_z1", "deep_reldec_z2", "mi_dqn_z2", "deep_reldec_zx", "mi_dqn_zx",
                    "augmented_max_avg_zx", "augmented_max_zx", "augmented_average_zx"}:
        # These are deep decoders
        decoder = dispatcher.get_decoder(method)
        return evaluate_deep_method(
            decoder=decoder,
            snr_db=float(snr_db),
            code_rate=float(code_rate),
            i_max=i_max,
            target_frame_errors=int(target_frame_errors),
            max_frames=int(max_frames),
            rng=rng,
            all_zero_only=bool(all_zero_only),
            method_name=method,
        )
    
    elif method in {"mi_tabular_z2", "mi_tabular_zx"}:
        # MI tabular decoders
        decoder = dispatcher.get_decoder(method)
        return evaluate_mi_tabular_method(
            decoder=decoder,
            snr_db=float(snr_db),
            code_rate=float(code_rate),
            i_max=i_max,
            target_frame_errors=int(target_frame_errors),
            max_frames=int(max_frames),
            rng=rng,
            all_zero_only=bool(all_zero_only),
            method_name=method,
        )
    
    elif method in {"mi_naive_z2", "mi_naive_zx"}:
        # MI naive decoders
        decoder = dispatcher.get_decoder(method)
        if not isinstance(decoder, MiReldecBaselineDecoder):
            raise TypeError(f"Expected MiReldecBaselineDecoder for {method}, got {type(decoder)}")
        return decoder.evaluate(
            snr_db=float(snr_db),
            code_rate=float(code_rate),
            i_max=i_max,
            target_frame_errors=int(target_frame_errors),
            max_frames=int(max_frames),
            rng=rng,
            all_zero_only=bool(all_zero_only),
            method_name=method,
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
