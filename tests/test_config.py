from __future__ import annotations

from pathlib import Path

import pytest

import preference_lab.config as config_module


def _config_yaml(
    *,
    seed: str = "7",
    train_data: str = "../data/preferences.jsonl",
    output_dir: str = "runs",
    method: str = "orpo",
    beta: str = "0.25",
    lambda_orpo: str = "0.0",
    max_length: str = "1024",
    batch_size: str = "8",
    regression_prompts: str = "prompts/regression.md",
    root_extra: str = "",
    paths_extra: str = "",
    training_extra: str = "",
    evaluation_extra: str = "",
) -> str:
    return (
        f"seed: {seed}\n"
        "paths:\n"
        f"  train_data: {train_data}\n"
        f"  output_dir: {output_dir}\n"
        f"{paths_extra}"
        "training:\n"
        f"  method: {method}\n"
        f"  beta: {beta}\n"
        f"  lambda_orpo: {lambda_orpo}\n"
        f"  max_length: {max_length}\n"
        f"  batch_size: {batch_size}\n"
        f"{training_extra}"
        "evaluation:\n"
        f"  regression_prompts: {regression_prompts}\n"
        f"{evaluation_extra}"
        f"{root_extra}"
    )


def _write_config(tmp_path: Path, document: str) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(document, encoding="utf-8")
    return config_path


def _assert_invalid_config(config_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid config") as exc_info:
        config_module.load_config(config_path)

    assert str(config_path) in str(exc_info.value)


def test_load_config_returns_typed_models_and_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _config_yaml())

    config = config_module.load_config(config_path)

    assert isinstance(config, config_module.LabConfig)
    assert isinstance(config.paths, config_module.PathsSettings)
    assert isinstance(config.training, config_module.TrainingSettings)
    assert isinstance(config.evaluation, config_module.EvaluationSettings)
    assert config.seed == 7
    assert config.paths.train_data == (tmp_path / "data/preferences.jsonl").resolve()
    assert config.paths.output_dir == (config_path.parent / "runs").resolve()
    assert config.training.method == "orpo"
    assert config.training.beta == 0.25
    assert config.training.lambda_orpo == 0.0
    assert config.training.max_length == 1024
    assert config.training.batch_size == 8
    assert config.evaluation.regression_prompts == (
        config_path.parent / "prompts/regression.md"
    ).resolve()


def test_load_config_preserves_absolute_paths(tmp_path: Path) -> None:
    train_data = (tmp_path / "external/train.jsonl").resolve()
    output_dir = (tmp_path / "external/output").resolve()
    prompts = (tmp_path / "external/prompts.md").resolve()
    config_path = _write_config(
        tmp_path,
        _config_yaml(
            train_data=str(train_data),
            output_dir=str(output_dir),
            regression_prompts=str(prompts),
        ),
    )

    config = config_module.load_config(config_path)

    assert config.paths.train_data == train_data
    assert config.paths.output_dir == output_dir
    assert config.evaluation.regression_prompts == prompts


def test_models_supply_established_scalar_defaults(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        "paths:\n"
        "  train_data: train.jsonl\n"
        "  output_dir: output\n"
        "training:\n"
        "  method: mock\n"
        "evaluation:\n"
        "  regression_prompts: prompts.md\n",
    )

    config = config_module.load_config(config_path)

    assert config.seed == 42
    assert config.training.beta == 0.1
    assert config.training.lambda_orpo == 0.1
    assert config.training.max_length == 512
    assert config.training.batch_size == 2


@pytest.mark.parametrize("document", ["", "- paths\n- training\n", "42\n", "paths: [\n"])
def test_load_config_rejects_invalid_yaml_documents(tmp_path: Path, document: str) -> None:
    config_path = _write_config(tmp_path, document)

    _assert_invalid_config(config_path)


@pytest.mark.parametrize(
    "document",
    [
        "seed: 7\ntraining:\n  method: dpo\nevaluation:\n  regression_prompts: prompts.md\n",
        _config_yaml(train_data="null"),
        _config_yaml(output_dir="null"),
    ],
)
def test_load_config_rejects_missing_paths(tmp_path: Path, document: str) -> None:
    config_path = _write_config(tmp_path, document)

    _assert_invalid_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_data", "123"),
        ("output_dir", "true"),
        ("regression_prompts", "1.5"),
    ],
)
def test_load_config_rejects_non_path_scalars(
    tmp_path: Path, field: str, value: str
) -> None:
    values = {
        "train_data": "train.jsonl",
        "output_dir": "output",
        "regression_prompts": "prompts.md",
    }
    values[field] = value
    config_path = _write_config(tmp_path, _config_yaml(**values))

    _assert_invalid_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "ipo"),
        ("beta", "0"),
        ("beta", "-0.1"),
        ("lambda_orpo", "-0.1"),
        ("max_length", "0"),
        ("batch_size", "-1"),
    ],
)
def test_load_config_rejects_invalid_training_values(
    tmp_path: Path, field: str, value: str
) -> None:
    values = {
        "method": "dpo",
        "beta": "0.1",
        "lambda_orpo": "0.1",
        "max_length": "512",
        "batch_size": "2",
    }
    values[field] = value
    config_path = _write_config(tmp_path, _config_yaml(**values))

    _assert_invalid_config(config_path)


@pytest.mark.parametrize(
    "document",
    [
        _config_yaml(root_extra="unexpected: true\n"),
        _config_yaml(paths_extra="  unexpected: true\n"),
        _config_yaml(training_extra="  unexpected: true\n"),
        _config_yaml(evaluation_extra="  unexpected: true\n"),
    ],
)
def test_load_config_forbids_unknown_fields(tmp_path: Path, document: str) -> None:
    config_path = _write_config(tmp_path, document)

    _assert_invalid_config(config_path)
