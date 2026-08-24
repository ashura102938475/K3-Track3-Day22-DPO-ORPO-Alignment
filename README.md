# Preference Alignment Lab: DPO \& ORPO Starter

Production-style skeleton for a 2-hour lab on preference alignment. The core data, loss, and
evaluation exercises are implemented; the optional trainer remains a `TODO(student)` exercise.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build and interpret a deterministic baseline for pairwise preference evaluation.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
uv python install
uv sync --locked --extra dev
uv run pytest -q
```

`uv python install` honors the version in `.python-version`, and `uv sync` creates and manages the
project environment. No manual virtual-environment activation is required.

To install the optional training dependencies:

```bash
uv sync --locked --extra dev --extra train
```

This installs the PyTorch/Transformers/TRL stack only. The optional
`PreferenceTrainer.train()` implementation is still a student exercise and is not invoked by the
evaluation command.

## Evaluation scope

`pref-lab evaluate` is a deterministic lexical-overlap baseline: it compares the unique words in
each prompt with the words in its chosen and rejected responses, then writes pairwise accuracy to
`outputs/metrics.json`. It does not load or evaluate a trained model, compare before/after training,
or run the configured regression prompts. The `outputs/` directory contains generated artifacts
and is ignored by Git.

## Lab rules

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless you have a clear reason.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `uv sync --locked --extra dev && uv run pytest -q` |
| 30-50 min | Implement dataset validation/collator | `uv run pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `uv run python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO loss TODOs | `uv run pytest tests/test_losses.py` |
| 100-115 min | Run the lexical baseline and write metrics | `uv run pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | One-minute demo | `uv run python -m json.tool outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Final verification

Run the same checks used by CI:

```bash
uv sync --locked --extra dev
uv run ruff check src tests
uv run mypy src
uv run pytest -q
uv run pref-lab validate data/sample_preferences.jsonl
uv run pref-lab evaluate --config configs/local.yaml
uv run python -m json.tool outputs/metrics.json
```

`bash scripts/smoke_test.sh` is a shorter end-to-end check; the script uses `uv run`, so it also
works from a fresh shell without activating `.venv`.

## Production checklist

- [x] Dataset schema validated.
- [x] Train/eval split utility groups by prompt, not by row.
- [x] Config committed; generated artifacts ignored.
- [x] Metrics saved as JSON.
- [ ] Safety regression prompts run before/after training.
- [x] Data card updated.
