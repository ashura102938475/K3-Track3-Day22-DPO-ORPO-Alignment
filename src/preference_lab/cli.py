from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import JsonScalar, lexical_overlap_score, pairwise_metrics, write_metrics
from .schemas import PreferenceExample

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(
    data: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    examples = _load_examples(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    try:
        cfg = load_config(config)
    except (OSError, ValueError) as error:
        _abort(_concise_error(error))

    examples = _load_examples(cfg.paths.train_data)
    chosen_scores = [lexical_overlap_score(example.prompt, example.chosen) for example in examples]
    rejected_scores = [
        lexical_overlap_score(example.prompt, example.rejected) for example in examples
    ]
    metrics: dict[str, JsonScalar] = dict(
        pairwise_metrics(examples, chosen_scores, rejected_scores)
    )
    metrics.update(
        scorer="lexical_overlap_v1",
        evaluation_scope="full_sample_lexical_baseline",
    )
    try:
        out = write_metrics(metrics, cfg.paths.output_dir)
    except OSError as error:
        _abort(_concise_error(error))
    print(f"[green]Wrote metrics to {out}[/green]")


def _load_examples(path: Path) -> list[PreferenceExample]:
    try:
        examples = load_jsonl(path)
    except (OSError, ValueError) as error:
        _abort(_concise_error(error))
    if not examples:
        _abort(f"{path}: no preference examples found")
    return examples


def _concise_error(error: Exception) -> str:
    return str(error).splitlines()[0]


def _abort(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
