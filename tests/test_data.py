import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")

    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_json_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        load_jsonl(bad)


def test_schema_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad-schema.jsonl"
    bad.write_text(
        '{"prompt":"a","chosen":"same","rejected":"different"}\n'
        '{"prompt":"b","chosen":"same","rejected":" SAME "}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"bad-schema\.jsonl:2: invalid schema"):
        load_jsonl(bad)


def test_loader_rejects_normalized_duplicate_prompts(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"prompt":"What is DPO?","chosen":"a","rejected":"b"}\n'
        '{"prompt":" what   IS dpo? ","chosen":"c","rejected":"d"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate\.jsonl:2: duplicate prompt.*line 1"):
        load_jsonl(duplicate)


def test_loader_rejects_unicode_canonical_duplicate_prompts(tmp_path: Path) -> None:
    duplicate = tmp_path / "unicode-duplicate.jsonl"
    duplicate.write_text(
        '{"prompt":"Caf\u00e9","chosen":"a","rejected":"b"}\n'
        '{"prompt":"Cafe\u0301","chosen":"c","rejected":"d"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unicode-duplicate\.jsonl:2: duplicate prompt"):
        load_jsonl(duplicate)


def test_schema_rejects_responses_that_only_differ_in_case_and_spacing() -> None:
    with pytest.raises(ValidationError, match="chosen and rejected must differ"):
        PreferenceExample(
            prompt="prompt",
            chosen="  Same   answer ",
            rejected="same answer",
        )


def test_schema_rejects_unicode_equivalent_responses() -> None:
    with pytest.raises(ValidationError, match="chosen and rejected must differ"):
        PreferenceExample(
            prompt="prompt",
            chosen="Caf\u00e9",
            rejected="Cafe\u0301",
        )


def test_schema_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PreferenceExample.model_validate(
            {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
                "unexpected": True,
            }
        )


def test_schema_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        PreferenceExample(prompt="   ", chosen="a", rejected="b")


@pytest.mark.parametrize(
    "validation_ratio",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_split_rejects_invalid_validation_ratios(validation_ratio: float) -> None:
    examples = [PreferenceExample(prompt="p1", chosen="a", rejected="b")]

    with pytest.raises(ValueError, match="validation_ratio"):
        split_by_prompt(examples, validation_ratio=validation_ratio)


@pytest.mark.parametrize("seed", [None, True, False, 7.0, "7", 7 + 0j])
def test_split_rejects_seed_that_is_not_an_integer(seed: object) -> None:
    examples = [PreferenceExample(prompt="p1", chosen="a", rejected="b")]

    with pytest.raises(ValueError, match="seed must be an integer"):
        split_by_prompt(examples, seed=seed)  # type: ignore[arg-type]


def test_split_ratio_zero_returns_every_row_in_train() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="e", rejected="f"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0, seed=7)

    assert train == examples
    assert validation == []


def test_split_ratio_one_returns_every_row_in_validation() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="e", rejected="f"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=1, seed=7)

    assert train == []
    assert validation == examples


def test_split_empty_input_returns_two_empty_partitions() -> None:
    assert split_by_prompt([], validation_ratio=0.5, seed=7) == ([], [])


def test_split_single_normalized_prompt_group_stays_in_train() -> None:
    examples = [
        PreferenceExample(prompt="Topic", chosen="a", rejected="b"),
        PreferenceExample(prompt=" topic ", chosen="c", rejected="d"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert train == examples
    assert validation == []


def test_split_prompt_membership_is_stable_under_input_reordering() -> None:
    examples = [
        PreferenceExample(prompt="alpha", chosen="a", rejected="b"),
        PreferenceExample(prompt="beta", chosen="c", rejected="d"),
        PreferenceExample(prompt="gamma", chosen="e", rejected="f"),
        PreferenceExample(prompt="delta", chosen="g", rejected="h"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=0)
    reordered_train, reordered_validation = split_by_prompt(
        list(reversed(examples)), validation_ratio=0.5, seed=0
    )

    assert {example.prompt for example in train} == {example.prompt for example in reordered_train}
    assert {example.prompt for example in validation} == {
        example.prompt for example in reordered_validation
    }


def test_split_is_deterministic_for_a_seed() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
        PreferenceExample(prompt="p3", chosen="e", rejected="f"),
    ]

    first = split_by_prompt(examples, validation_ratio=0.5, seed=7)
    second = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert first == second


def test_split_membership_is_stable_across_python_hash_seeds() -> None:
    script = """
import json

from preference_lab.data import split_by_prompt
from preference_lab.schemas import PreferenceExample

examples = [
    PreferenceExample(prompt=prompt, chosen="a", rejected="b")
    for prompt in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
]
train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=7)
print(json.dumps({
    "train": sorted(example.prompt for example in train),
    "validation": sorted(example.prompt for example in validation),
}, sort_keys=True))
"""
    memberships: list[str] = []
    for hash_seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        memberships.append(completed.stdout.strip())

    assert memberships[0] == memberships[1]


def test_split_conserves_every_input_row_exactly_once() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b", metadata={"row": 1}),
        PreferenceExample(prompt="p1", chosen="c", rejected="d", metadata={"row": 2}),
        PreferenceExample(prompt="p2", chosen="e", rejected="f", metadata={"row": 3}),
        PreferenceExample(prompt="p3", chosen="g", rejected="h", metadata={"row": 4}),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=7)

    assert sorted(example.metadata["row"] for example in train + validation) == [1, 2, 3, 4]


def test_split_has_no_leakage_between_unicode_normalized_prompt_groups() -> None:
    examples = [
        PreferenceExample(
            prompt="Caf\u00e9", chosen="a", rejected="b", metadata={"row": "composed"}
        ),
        PreferenceExample(
            prompt=" cafe\u0301 ", chosen="c", rejected="d", metadata={"row": "decomposed"}
        ),
        PreferenceExample(prompt="alpha", chosen="e", rejected="f", metadata={"row": "alpha"}),
        PreferenceExample(prompt="p3", chosen="g", rejected="h"),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=0)

    train_rows = {example.metadata.get("row") for example in train}
    validation_rows = {example.metadata.get("row") for example in validation}
    canonical_rows = {"composed", "decomposed"}
    assert canonical_rows <= train_rows or canonical_rows <= validation_rows


def test_split_preserves_original_row_order_within_each_partition() -> None:
    examples = [
        PreferenceExample(prompt="delta", chosen="a", rejected="b", metadata={"row": 0}),
        PreferenceExample(prompt="alpha", chosen="c", rejected="d", metadata={"row": 1}),
        PreferenceExample(prompt="charlie", chosen="e", rejected="f", metadata={"row": 2}),
        PreferenceExample(prompt="delta", chosen="g", rejected="h", metadata={"row": 3}),
        PreferenceExample(prompt="bravo", chosen="i", rejected="j", metadata={"row": 4}),
        PreferenceExample(prompt="alpha", chosen="k", rejected="l", metadata={"row": 5}),
    ]

    train, validation = split_by_prompt(examples, validation_ratio=0.5, seed=0)

    assert [example.metadata["row"] for example in train] == [1, 2, 5]
    assert [example.metadata["row"] for example in validation] == [0, 3, 4]
