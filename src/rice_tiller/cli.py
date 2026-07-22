from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from rice_tiller.exceptions import RiceTillerError

app = typer.Typer(no_args_is_help=True, help="Analyse 3D rice point clouds with DGCNN.")


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("outputs"), "--output", "-o"),
    scan_index: Optional[int] = typer.Option(None, "--scan-index"),
) -> None:
    """Analyse one preprocessed PLY or E57 point cloud."""
    from rice_tiller.pipeline import run_pipeline

    try:
        result = run_pipeline(
            input_path,
            config_path=config,
            output_root=output,
            scan_index=scan_index,
        )
    except (RiceTillerError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.summary_dict(), indent=2, ensure_ascii=False))


@app.command("analyze-batch")
def analyze_batch(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    pattern: str = typer.Option("*.ply", "--pattern"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("outputs"), "--output", "-o"),
) -> None:
    """Analyse all matching files in a directory."""
    from rice_tiller.pipeline import run_pipeline

    files = sorted(input_dir.glob(pattern))
    if not files:
        typer.echo(f"No files matched {pattern!r} in {input_dir}", err=True)
        raise typer.Exit(code=1)
    failures = 0
    for path in files:
        typer.echo(f"Analysing {path.name}...")
        try:
            result = run_pipeline(path, config_path=config, output_root=output)
            typer.echo(f"  {result.status}: {result.output_dir}")
        except (RiceTillerError, RuntimeError, ValueError) as exc:
            failures += 1
            typer.echo(f"  failed: {exc}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command("inspect-weight")
def inspect_weight(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True),
    metadata: Path = typer.Option(..., "--metadata", exists=True, readable=True),
) -> None:
    """Inspect checkpoint size and validate the model metadata."""
    from rice_tiller.models.checkpoint import inspect_checkpoint, load_model_metadata

    try:
        info = inspect_checkpoint(checkpoint)
        info["metadata"] = load_model_metadata(metadata)
    except RiceTillerError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(info, indent=2, ensure_ascii=False, default=str))

