from __future__ import annotations

import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: ValidationInfo) -> str:
        chosen = info.data.get("chosen")
        if isinstance(chosen, str) and normalize_text(chosen) == normalize_text(rejected):
            raise ValueError("chosen and rejected must differ")
        return rejected
