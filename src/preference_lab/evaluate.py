from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .schemas import PreferenceExample, normalize_text

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
JsonScalar = str | int | float | bool | None


def lexical_overlap_score(prompt: str, response: str) -> float:
    """Score topical relevance by unique prompt-token coverage."""
    prompt_tokens = set(_WORD_PATTERN.findall(normalize_text(prompt)))
    if not prompt_tokens:
        return 0.0
    response_tokens = set(_WORD_PATTERN.findall(normalize_text(response)))
    return len(prompt_tokens & response_tokens) / len(prompt_tokens)


def pairwise_metrics(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> dict[str, int | float]:
    """Return validated pairwise comparison metrics."""
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("score lengths must match examples")
    if not examples:
        raise ValueError("at least one example is required")

    for scores_name, scores in (
        ("chosen_scores", chosen_scores),
        ("rejected_scores", rejected_scores),
    ):
        for index, score in enumerate(scores):
            if not math.isfinite(score):
                raise ValueError(f"{scores_name}[{index}] must be finite")

    score_pairs = list(zip(chosen_scores, rejected_scores, strict=True))
    margins: list[float] = []
    for index, (chosen, rejected) in enumerate(score_pairs):
        margin = chosen - rejected
        if not math.isfinite(margin):
            raise ValueError(f"score margin[{index}] must be finite")
        margins.append(margin)

    wins = sum(chosen > rejected for chosen, rejected in score_pairs)
    losses = sum(chosen < rejected for chosen, rejected in score_pairs)
    ties = len(examples) - wins - losses
    return {
        "pairwise_accuracy": (wins + 0.5 * ties) / len(examples),
        "num_examples": len(examples),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "tie_rate": ties / len(examples),
        "mean_score_margin": math.fsum(margin / len(examples) for margin in margins),
    }


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return pairwise accuracy from the validated metric breakdown."""
    return float(pairwise_metrics(examples, chosen_scores, rejected_scores)["pairwise_accuracy"])


def write_metrics(metrics: dict[str, JsonScalar], output_dir: str | Path) -> Path:
    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric '{name}' must be finite")

    payload = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(payload, encoding="utf-8")
    return out
