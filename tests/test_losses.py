import sys
import warnings

import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def _valid_dpo_inputs() -> list[np.ndarray]:
    return [
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
    ]


def test_dpo_loss_matches_closed_form() -> None:
    loss = dpo_loss(*_valid_dpo_inputs(), beta=0.1)

    assert loss == pytest.approx(0.663597, abs=1e-5)


def test_dpo_loss_uses_the_mean_across_examples() -> None:
    loss = dpo_loss(
        np.array([-0.2, -0.4]),
        np.array([-1.2, -0.8]),
        np.array([-0.5, -0.3]),
        np.array([-1.0, -0.6]),
        beta=2.0,
    )

    assert loss == pytest.approx(0.45570027844990735)


def test_dpo_policy_equal_to_reference_is_log_two() -> None:
    chosen = np.array([-0.1, -3.0, -100.0])
    rejected = np.array([-2.0, -4.0, -50.0])

    loss = dpo_loss(chosen, rejected, chosen.copy(), rejected.copy(), beta=37.0)

    assert loss == pytest.approx(np.log(2.0))


def test_dpo_loss_decreases_as_margin_grows() -> None:
    ref = (np.array([-0.6]), np.array([-1.0]))

    weak = dpo_loss(np.array([-0.9]), np.array([-1.0]), *ref, beta=0.1)
    strong = dpo_loss(np.array([-0.1]), np.array([-3.0]), *ref, beta=0.1)

    assert strong < weak


def test_dpo_loss_is_stable_for_extreme_inputs() -> None:
    loss = dpo_loss(
        np.array([-1e4]),
        np.array([-1.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=1.0,
    )

    assert np.isfinite(loss)


def test_dpo_averages_repeated_maximum_finite_losses_without_overflow() -> None:
    maximum = sys.float_info.max

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        loss = dpo_loss(
            np.array([-maximum, -maximum]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            beta=1.0,
        )

    assert loss == maximum


def test_dpo_rejects_an_overflowed_margin_without_a_runtime_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="DPO margins must be finite"):
            dpo_loss(
                np.array([0.0]),
                np.array([-1e308]),
                np.array([-1e308]),
                np.array([0.0]),
                beta=1.0,
            )


def test_dpo_converts_inputs_to_float64_before_arithmetic() -> None:
    loss = dpo_loss(
        np.array([0], dtype=np.int8),
        np.array([-128], dtype=np.int8),
        np.array([0], dtype=np.int8),
        np.array([0], dtype=np.int8),
        beta=1.0,
    )

    assert loss == pytest.approx(2.572209372642415e-56)


def test_dpo_rejects_broadcastable_but_different_shapes() -> None:
    with pytest.raises(ValueError, match="exactly the same shape"):
        dpo_loss(
            np.full((2, 1), -0.5),
            np.full((1, 2), -1.5),
            np.full((2, 1), -0.6),
            np.full((2, 1), -1.0),
            beta=0.1,
        )


@pytest.mark.parametrize("empty_index", range(4))
def test_dpo_rejects_an_empty_log_probability_array(empty_index: int) -> None:
    inputs = _valid_dpo_inputs()
    inputs[empty_index] = np.array([])

    with pytest.raises(ValueError, match="non-empty"):
        dpo_loss(*inputs, beta=0.1)


@pytest.mark.parametrize("array_index", range(4))
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_dpo_rejects_nonfinite_log_probabilities(array_index: int, nonfinite: float) -> None:
    inputs = _valid_dpo_inputs()
    inputs[array_index] = np.array([nonfinite])

    with pytest.raises(ValueError, match="finite"):
        dpo_loss(*inputs, beta=0.1)


@pytest.mark.parametrize("array_index", range(4))
def test_dpo_rejects_complex_log_probabilities_without_a_warning(array_index: int) -> None:
    inputs = _valid_dpo_inputs()
    inputs[array_index] = np.array([-0.5 + 0.25j])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numeric array"):
            dpo_loss(*inputs, beta=0.1)


@pytest.mark.parametrize("array_index", range(4))
def test_dpo_rejects_positive_log_probabilities(array_index: int) -> None:
    inputs = _valid_dpo_inputs()
    inputs[array_index] = np.array([0.01])

    with pytest.raises(ValueError, match=r"<= 0"):
        dpo_loss(*inputs, beta=0.1)


@pytest.mark.parametrize("beta", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_dpo_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match=r"beta.*finite.*> 0"):
        dpo_loss(*_valid_dpo_inputs(), beta=beta)


@pytest.mark.parametrize(
    "beta",
    [True, False, 1 + 0j, "0.1", [0.1], np.array(0.1)],
)
def test_dpo_rejects_beta_that_is_not_a_real_scalar(beta: object) -> None:
    with pytest.raises(ValueError, match=r"beta.*real scalar"):
        dpo_loss(*_valid_dpo_inputs(), beta=beta)  # type: ignore[arg-type]


def _valid_orpo_inputs() -> list[np.ndarray]:
    return [
        np.array([1.0]),
        np.array([-0.5]),
        np.array([-1.5]),
    ]


def test_orpo_loss_matches_closed_form() -> None:
    loss = orpo_loss(*_valid_orpo_inputs(), lambda_orpo=0.1)

    assert loss == pytest.approx(1.017086, abs=1e-5)


def test_orpo_uses_separate_means_for_sft_and_preference_examples() -> None:
    loss = orpo_loss(
        np.array([1.0, 2.0, 3.0]),
        np.log(np.array([0.8, 0.5])),
        np.log(np.array([0.2, 0.25])),
        lambda_orpo=0.5,
    )

    assert loss == pytest.approx(2.087076673567054)


def test_orpo_tail_log_probabilities_retain_preference_direction() -> None:
    dispreferred = orpo_loss(
        np.array([0.0]),
        np.array([-100.0]),
        np.array([-50.0]),
        lambda_orpo=1.0,
    )
    preferred = orpo_loss(
        np.array([0.0]),
        np.array([-50.0]),
        np.array([-100.0]),
        lambda_orpo=1.0,
    )

    assert dispreferred == pytest.approx(50.0, abs=1e-12)
    assert preferred == pytest.approx(1.9287498479639178e-22, rel=1e-12, abs=0.0)
    assert dispreferred > np.log(2.0) > preferred


def test_orpo_is_finite_for_a_log_probability_next_to_zero() -> None:
    closest_negative_to_zero = np.nextafter(np.float64(0.0), np.float64(-1.0))

    loss = orpo_loss(
        np.array([0.0]),
        np.array([closest_negative_to_zero]),
        np.array([-1.0]),
        lambda_orpo=1.0,
    )

    assert np.isfinite(loss)
    assert loss < 1e-300


def test_orpo_allows_zero_lambda() -> None:
    loss = orpo_loss(
        np.array([1.0, 3.0]),
        np.array([-0.5]),
        np.array([-1.5]),
        lambda_orpo=0.0,
    )

    assert loss == pytest.approx(2.0)


def test_orpo_averages_repeated_maximum_finite_sft_losses_without_overflow() -> None:
    maximum = sys.float_info.max

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        loss = orpo_loss(
            np.array([maximum, maximum]),
            np.array([-0.5]),
            np.array([-1.5]),
            lambda_orpo=0.0,
        )

    assert loss == maximum


def test_orpo_rejects_an_overflowed_combined_loss_without_a_runtime_warning() -> None:
    maximum = sys.float_info.max

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="ORPO loss must be finite"):
            orpo_loss(
                np.array([maximum]),
                np.array([-1.5]),
                np.array([-0.5]),
                lambda_orpo=maximum,
            )


def test_orpo_rejects_broadcastable_but_different_preference_shapes() -> None:
    with pytest.raises(ValueError, match="exactly the same shape"):
        orpo_loss(
            np.array([1.0]),
            np.full((2, 1), -0.5),
            np.full((1, 2), -1.5),
            lambda_orpo=0.1,
        )


@pytest.mark.parametrize("empty_index", [1, 2])
def test_orpo_rejects_an_empty_log_probability_array(empty_index: int) -> None:
    inputs = _valid_orpo_inputs()
    inputs[empty_index] = np.array([])

    with pytest.raises(ValueError, match="non-empty"):
        orpo_loss(*inputs, lambda_orpo=0.1)


@pytest.mark.parametrize("array_index", [1, 2])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_orpo_rejects_nonfinite_log_probabilities(
    array_index: int, nonfinite: float
) -> None:
    inputs = _valid_orpo_inputs()
    inputs[array_index] = np.array([nonfinite])

    with pytest.raises(ValueError, match="finite"):
        orpo_loss(*inputs, lambda_orpo=0.1)


@pytest.mark.parametrize("array_index", range(3))
def test_orpo_rejects_complex_inputs_without_a_warning(array_index: int) -> None:
    inputs = _valid_orpo_inputs()
    inputs[array_index] = np.array([-0.5 + 0.25j])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numeric array"):
            orpo_loss(*inputs, lambda_orpo=0.1)


@pytest.mark.parametrize("array_index", [1, 2])
@pytest.mark.parametrize("invalid_logp", [0.0, 0.01])
def test_orpo_rejects_nonnegative_log_probabilities(
    array_index: int, invalid_logp: float
) -> None:
    inputs = _valid_orpo_inputs()
    inputs[array_index] = np.array([invalid_logp])

    with pytest.raises(ValueError, match=r"strictly < 0"):
        orpo_loss(*inputs, lambda_orpo=0.1)


def test_orpo_rejects_empty_sft_nll() -> None:
    with pytest.raises(ValueError, match=r"sft_nll.*non-empty"):
        orpo_loss(
            np.array([]),
            np.array([-0.5]),
            np.array([-1.5]),
            lambda_orpo=0.1,
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_orpo_rejects_nonfinite_sft_nll(nonfinite: float) -> None:
    with pytest.raises(ValueError, match=r"sft_nll.*finite"):
        orpo_loss(
            np.array([nonfinite]),
            np.array([-0.5]),
            np.array([-1.5]),
            lambda_orpo=0.1,
        )


def test_orpo_rejects_negative_sft_nll() -> None:
    with pytest.raises(ValueError, match=r"sft_nll.*>= 0"):
        orpo_loss(
            np.array([-0.01]),
            np.array([-0.5]),
            np.array([-1.5]),
            lambda_orpo=0.1,
        )


@pytest.mark.parametrize("lambda_orpo", [-0.1, np.nan, np.inf, -np.inf])
def test_orpo_rejects_invalid_lambda(lambda_orpo: float) -> None:
    with pytest.raises(ValueError, match=r"lambda_orpo.*finite.*>= 0"):
        orpo_loss(*_valid_orpo_inputs(), lambda_orpo=lambda_orpo)


@pytest.mark.parametrize(
    "lambda_orpo",
    [True, False, 1 + 0j, "0.1", [0.1], np.array(0.1)],
)
def test_orpo_rejects_lambda_that_is_not_a_real_scalar(lambda_orpo: object) -> None:
    with pytest.raises(ValueError, match=r"lambda_orpo.*real scalar"):
        orpo_loss(
            *_valid_orpo_inputs(),
            lambda_orpo=lambda_orpo,  # type: ignore[arg-type]
        )
