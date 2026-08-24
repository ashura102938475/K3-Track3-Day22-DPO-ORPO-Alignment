.PHONY: setup test lint typecheck format run-eval clean

setup:
	uv sync --locked --extra dev

test:
	uv run pytest -q tests

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

format:
	uv run ruff format src tests

run-eval:
	uv run pref-lab evaluate --config configs/local.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs
