from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_LOG_HALF = -np.log(2.0)


def _as_float64_array(values: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real numeric array.")
    try:
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc


def _as_real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar.")  # noqa: TRY004
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar.") from exc


def _log_sigmoid(values: FloatArray) -> FloatArray:
    return np.asarray(-np.logaddexp(0.0, -values), dtype=np.float64)


def _stable_nonnegative_mean(values: FloatArray) -> float:
    largest = float(np.max(values))
    if largest == 0.0:
        return 0.0
    return float(largest * np.mean(values / largest))


def dpo_loss(
    policy_chosen_logps: ArrayLike,
    policy_rejected_logps: ArrayLike,
    ref_chosen_logps: ArrayLike,
    ref_rejected_logps: ArrayLike,
    beta: float,
) -> float:
    """Compute the mean, numerically stable DPO loss for a batch."""
    arrays = (
        _as_float64_array(policy_chosen_logps, "policy_chosen_logps"),
        _as_float64_array(policy_rejected_logps, "policy_rejected_logps"),
        _as_float64_array(ref_chosen_logps, "ref_chosen_logps"),
        _as_float64_array(ref_rejected_logps, "ref_rejected_logps"),
    )
    if any(values.size == 0 for values in arrays):
        raise ValueError("DPO log-probability arrays must be non-empty.")
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        shapes = ", ".join(str(values.shape) for values in arrays)
        raise ValueError(
            "DPO log-probability arrays must have exactly the same shape; "
            f"got {shapes}."
        )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("DPO log-probability arrays must contain only finite values.")
    if any(np.any(values > 0.0) for values in arrays):
        raise ValueError("DPO log-probability arrays must contain values <= 0.")

    beta_value = _as_real_scalar(beta, "beta")
    if not np.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("beta must be finite and > 0.")

    policy_chosen, policy_rejected, reference_chosen, reference_rejected = arrays
    with np.errstate(over="ignore", invalid="ignore"):
        policy_log_ratios = policy_chosen - policy_rejected
        reference_log_ratios = reference_chosen - reference_rejected
        margins = beta_value * (policy_log_ratios - reference_log_ratios)
    if not np.all(np.isfinite(margins)):
        raise ValueError(
            "DPO margins must be finite; log-probability differences or beta are too large."
        )
    return _stable_nonnegative_mean(-_log_sigmoid(margins))


def _log_one_minus_exp(log_probabilities: FloatArray) -> FloatArray:
    result = np.empty_like(log_probabilities)
    near_zero = log_probabilities > _LOG_HALF
    result[near_zero] = np.log(-np.expm1(log_probabilities[near_zero]))
    result[~near_zero] = np.log1p(-np.exp(log_probabilities[~near_zero]))
    return result


def _log_odds(log_probabilities: FloatArray) -> FloatArray:
    return np.asarray(
        log_probabilities - _log_one_minus_exp(log_probabilities), dtype=np.float64
    )


def orpo_loss(
    sft_nll: ArrayLike,
    chosen_logps: ArrayLike,
    rejected_logps: ArrayLike,
    lambda_orpo: float,
) -> float:
    """Compute mean SFT NLL plus a stable odds-ratio preference penalty."""
    sft_values = _as_float64_array(sft_nll, "sft_nll")
    chosen_values = _as_float64_array(chosen_logps, "chosen_logps")
    rejected_values = _as_float64_array(rejected_logps, "rejected_logps")

    if chosen_values.size == 0 or rejected_values.size == 0:
        raise ValueError("ORPO chosen and rejected log-probability arrays must be non-empty.")
    if chosen_values.shape != rejected_values.shape:
        raise ValueError(
            "ORPO chosen and rejected log-probability arrays must have exactly the same "
            f"shape; got {chosen_values.shape} and {rejected_values.shape}."
        )
    if not np.all(np.isfinite(chosen_values)) or not np.all(np.isfinite(rejected_values)):
        raise ValueError(
            "ORPO chosen and rejected log-probability arrays must contain only finite values."
        )
    if np.any(chosen_values >= 0.0) or np.any(rejected_values >= 0.0):
        raise ValueError(
            "ORPO chosen and rejected log-probability arrays must contain values strictly < 0."
        )

    if sft_values.size == 0:
        raise ValueError("sft_nll must be non-empty.")
    if not np.all(np.isfinite(sft_values)):
        raise ValueError("sft_nll must contain only finite values.")
    if np.any(sft_values < 0.0):
        raise ValueError("sft_nll must contain values >= 0.")

    lambda_value = _as_real_scalar(lambda_orpo, "lambda_orpo")
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("lambda_orpo must be finite and >= 0.")

    log_odds_difference = _log_odds(chosen_values) - _log_odds(rejected_values)
    preference_penalty = -_log_sigmoid(log_odds_difference)
    sft_mean = _stable_nonnegative_mean(sft_values)
    preference_mean = _stable_nonnegative_mean(preference_penalty)
    with np.errstate(over="ignore", invalid="ignore"):
        loss = sft_mean + lambda_value * preference_mean
    if not np.isfinite(loss):
        raise ValueError("ORPO loss must be finite; sft_nll or lambda_orpo is too large.")
    return float(loss)
