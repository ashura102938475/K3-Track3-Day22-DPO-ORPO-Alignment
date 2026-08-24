import re
import sys
from pathlib import Path

import pytest

import preference_lab.evaluate as evaluate_module
from preference_lab.schemas import PreferenceExample


def _example(prompt: str = "p") -> PreferenceExample:
    return PreferenceExample(prompt=prompt, chosen="a", rejected="b")


def test_pairwise_metrics_reports_complete_breakdown() -> None:
    examples = [_example("win"), _example("loss"), _example("tie")]

    metrics = evaluate_module.pairwise_metrics(
        examples,
        [2.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    )

    assert metrics == {
        "losses": 1,
        "mean_score_margin": 0.0,
        "num_examples": 3,
        "pairwise_accuracy": 0.5,
        "tie_rate": 1 / 3,
        "ties": 1,
        "wins": 1,
    }


@pytest.mark.parametrize(
    ("chosen_scores", "rejected_scores"),
    [
        ([], [1.0]),
        ([1.0], []),
        ([1.0, 2.0], [1.0]),
        ([1.0], [1.0, 2.0]),
    ],
)
def test_pairwise_metrics_rejects_score_length_mismatch(
    chosen_scores: list[float], rejected_scores: list[float]
) -> None:
    with pytest.raises(ValueError, match="score lengths must match examples"):
        evaluate_module.pairwise_metrics([_example()], chosen_scores, rejected_scores)


def test_pairwise_metrics_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        evaluate_module.pairwise_metrics([], [], [])


@pytest.mark.parametrize(
    ("chosen_scores", "rejected_scores", "bad_input"),
    [
        ([float("nan")], [0.0], "chosen_scores[0]"),
        ([float("inf")], [0.0], "chosen_scores[0]"),
        ([float("-inf")], [0.0], "chosen_scores[0]"),
        ([0.0], [float("nan")], "rejected_scores[0]"),
        ([0.0], [float("inf")], "rejected_scores[0]"),
        ([0.0], [float("-inf")], "rejected_scores[0]"),
    ],
)
def test_pairwise_metrics_rejects_non_finite_scores(
    chosen_scores: list[float], rejected_scores: list[float], bad_input: str
) -> None:
    with pytest.raises(ValueError, match=rf"{re.escape(bad_input)} must be finite"):
        evaluate_module.pairwise_metrics([_example()], chosen_scores, rejected_scores)


def test_pairwise_metrics_rejects_non_finite_computed_margin() -> None:
    max_score = sys.float_info.max

    with pytest.raises(ValueError, match=r"score margin\[0\] must be finite"):
        evaluate_module.pairwise_metrics([_example()], [max_score], [-max_score])


def test_pairwise_metrics_averages_large_finite_margins_without_overflow() -> None:
    max_score = sys.float_info.max

    metrics = evaluate_module.pairwise_metrics(
        [_example("first"), _example("second")],
        [max_score, max_score],
        [0.0, 0.0],
    )

    assert metrics["mean_score_margin"] == max_score


def test_pairwise_accuracy_uses_validated_metrics_contract() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        evaluate_module.pairwise_accuracy([], [], [])


def test_lexical_overlap_score_reads_prompt_and_response_content() -> None:
    prompt = "Explain self attention in transformers"

    relevant = evaluate_module.lexical_overlap_score(
        prompt, "Transformers use self attention across tokens."
    )
    unrelated = evaluate_module.lexical_overlap_score(prompt, "A recipe needs flour and water.")

    assert relevant > unrelated


def test_lexical_overlap_score_normalizes_unicode_case_and_whitespace() -> None:
    score = evaluate_module.lexical_overlap_score(
        "  CAF\u00c9\tMODEL  ",
        "A cafe\u0301 model",
    )

    assert score == 1.0


def test_write_metrics_accepts_json_scalar_metadata_and_creates_sorted_strict_json(
    tmp_path: Path,
) -> None:
    metrics: dict[str, str | int | float | bool | None] = {
        "scorer": "lexical_overlap_v1",
        "num_examples": 3,
        "pairwise_accuracy": 0.5,
        "is_baseline": True,
        "notes": None,
    }

    output = evaluate_module.write_metrics(metrics, tmp_path)

    assert output == tmp_path / "metrics.json"
    assert output.read_text(encoding="utf-8") == (
        "{\n"
        '  "is_baseline": true,\n'
        '  "notes": null,\n'
        '  "num_examples": 3,\n'
        '  "pairwise_accuracy": 0.5,\n'
        '  "scorer": "lexical_overlap_v1"\n'
        "}"
    )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_write_metrics_rejects_non_finite_values_before_creating_output_directory(
    tmp_path: Path, non_finite: float
) -> None:
    output_dir = tmp_path / "not-created"

    with pytest.raises(ValueError, match="metric 'bad_metric' must be finite"):
        evaluate_module.write_metrics({"bad_metric": non_finite}, output_dir)

    assert not output_dir.exists()
