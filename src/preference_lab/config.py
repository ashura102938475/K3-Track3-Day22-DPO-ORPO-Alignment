from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsSettings(_SettingsModel):
    train_data: Path
    output_dir: Path


class TrainingSettings(_SettingsModel):
    method: Literal["dpo", "orpo", "mock"]
    beta: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    lambda_orpo: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=2, gt=0)


class EvaluationSettings(_SettingsModel):
    regression_prompts: Path


class LabConfig(_SettingsModel):
    seed: int = 42
    paths: PathsSettings
    training: TrainingSettings
    evaluation: EvaluationSettings


def _resolve_path(path: Path, config_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _resolve_paths(config: LabConfig, config_dir: Path) -> LabConfig:
    return config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "train_data": _resolve_path(config.paths.train_data, config_dir),
                    "output_dir": _resolve_path(config.paths.output_dir, config_dir),
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={
                    "regression_prompts": _resolve_path(
                        config.evaluation.regression_prompts, config_dir
                    )
                }
            ),
        }
    )


def load_config(path: str | Path) -> LabConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            document: object = yaml.safe_load(file)
        config = LabConfig.model_validate(document)
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"{config_path}: invalid config: {error}") from error

    return _resolve_paths(config, config_path.parent.resolve())
