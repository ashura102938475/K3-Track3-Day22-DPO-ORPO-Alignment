from __future__ import annotations

import json
import math
import random
from numbers import Integral
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample, normalize_text


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load validated, uniquely prompted preference examples from JSONL."""
    source = Path(path)
    examples: list[PreferenceExample] = []
    prompt_lines: dict[str, int] = {}
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON - {exc.msg}") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{source}:{line_no}: invalid schema - {exc}") from exc

            prompt_key = normalize_text(example.prompt)
            if prompt_key in prompt_lines:
                first_line = prompt_lines[prompt_key]
                raise ValueError(
                    f"{source}:{line_no}: duplicate prompt (first seen on line {first_line})"
                )
            prompt_lines[prompt_key] = line_no
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split whole normalized-prompt groups with deterministic shuffling."""
    if not math.isfinite(validation_ratio) or not 0 <= validation_ratio <= 1:
        raise ValueError("validation_ratio must be finite and between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")  # noqa: TRY004

    prompt_groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        prompt_groups.setdefault(normalize_text(example.prompt), []).append(example)
    prompt_keys = sorted(prompt_groups)
    random.Random(int(seed)).shuffle(prompt_keys)

    cut = int(len(prompt_keys) * (1 - validation_ratio))
    if 0 < validation_ratio < 1:
        if len(prompt_keys) == 1:
            cut = 1
        elif len(prompt_keys) > 1:
            cut = min(max(cut, 1), len(prompt_keys) - 1)
    train_prompts = set(prompt_keys[:cut])

    train = [example for example in examples if normalize_text(example.prompt) in train_prompts]
    validation = [
        example for example in examples if normalize_text(example.prompt) not in train_prompts
    ]
    return train, validation
