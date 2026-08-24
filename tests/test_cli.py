from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from preference_lab.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data/sample_preferences.jsonl"


def _write_config(tmp_path: Path, train_data: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "outputs"
    regression_prompts = tmp_path / "regression-prompts.md"
    regression_prompts.write_text("# Prompts\n", encoding="utf-8")
    config = tmp_path / "local.yaml"
    config.write_text(
        "paths:\n"
        f"  train_data: {json.dumps(str(train_data))}\n"
        f"  output_dir: {json.dumps(str(output_dir))}\n"
        "training:\n"
        "  method: dpo\n"
        "evaluation:\n"
        f"  regression_prompts: {json.dumps(str(regression_prompts))}\n",
        encoding="utf-8",
    )
    return config, output_dir


def _assert_clean_error(result: Result, expected: str) -> None:
    assert result.exit_code != 0
    assert expected in result.output
    assert "Traceback" not in result.output


def test_evaluate_writes_exact_sample_breakdown_and_metadata(tmp_path: Path) -> None:
    config, output_dir = _write_config(tmp_path, SAMPLE_DATA)

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config)])

    assert result.exit_code == 0, result.output
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {
        "evaluation_scope": "full_sample_lexical_baseline",
        "losses": 7,
        "mean_score_margin": 0.015940656565656557,
        "num_examples": 24,
        "pairwise_accuracy": 0.5208333333333334,
        "scorer": "lexical_overlap_v1",
        "tie_rate": 0.375,
        "ties": 9,
        "wins": 8,
    }


def test_evaluate_reports_metrics_write_failure_cleanly(tmp_path: Path) -> None:
    config, output_dir = _write_config(tmp_path, SAMPLE_DATA)
    output_dir.write_text("already a file", encoding="utf-8")

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config)])

    _assert_clean_error(result, f"File exists: '{output_dir}'")


@pytest.mark.parametrize("command", ["validate", "evaluate"])
def test_commands_report_invalid_data_with_file_and_line_context(
    tmp_path: Path, command: str
) -> None:
    bad_data = tmp_path / "bad.jsonl"
    bad_data.write_text(
        '{"prompt":"p","chosen":"a","rejected":"b"}\n{oops\n',
        encoding="utf-8",
    )
    if command == "validate":
        arguments = ["validate", str(bad_data)]
    else:
        config, _ = _write_config(tmp_path, bad_data)
        arguments = ["evaluate", "--config", str(config)]

    result = CliRunner().invoke(app, arguments)

    _assert_clean_error(result, f"{bad_data}:2: invalid JSON")


@pytest.mark.parametrize("command", ["validate", "evaluate"])
def test_commands_reject_empty_data_cleanly(tmp_path: Path, command: str) -> None:
    empty_data = tmp_path / "empty.jsonl"
    empty_data.write_text("\n", encoding="utf-8")
    if command == "validate":
        arguments = ["validate", str(empty_data)]
    else:
        config, _ = _write_config(tmp_path, empty_data)
        arguments = ["evaluate", "--config", str(config)]

    result = CliRunner().invoke(app, arguments)

    _assert_clean_error(result, f"{empty_data}: no preference examples found")


def test_evaluate_reports_invalid_config_cleanly(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("{}\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config)])

    _assert_clean_error(result, f"{config}: invalid config")


@pytest.mark.parametrize(
    "arguments",
    [
        ["validate", "missing.jsonl"],
        ["evaluate", "--config", "missing.yaml"],
    ],
)
def test_cli_path_inputs_must_be_existing_files(arguments: list[str]) -> None:
    result = CliRunner().invoke(app, arguments)

    _assert_clean_error(result, "does not exist")
