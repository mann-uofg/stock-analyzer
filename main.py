#!/usr/bin/env python3
"""Local stock analyzer - CLI entry point.

    python main.py --ticker NVDA
    python main.py --ticker AAPL --no-llm --json out.json
    python main.py --ticker TSLA,AMD,SPY
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

from analyzer import cache as cache_mod
from analyzer import engine, llm, report

from analyzer.datafeed import DataError

app = typer.Typer(
    add_completion=False,
    help="Zero-cost, fully local stock analyzer with quantitative scoring "
         "and offline LLM synthesis.",
)
console = Console()


def _analyse_one(
    ticker: str, period: str, use_cache: bool, use_llm: bool,
    skip_options: bool, quiet: bool, numeric_authority: bool = True,
) -> dict[str, Any] | None:
    """Run one ticker end to end, returning the payload (with narrative)."""
    try:
        if quiet:
            payload = engine.analyse(ticker, period=period, use_cache=use_cache,
                                     skip_options=skip_options)
        else:
            with console.status(f"[cyan]Analysing {ticker}...") as status:
                payload = engine.analyse(
                    ticker, period=period, use_cache=use_cache, skip_options=skip_options,
                    progress=lambda msg: status.update(f"[cyan]{ticker}: {msg}..."),
                )
    except DataError as exc:
        console.print(Text(f"✖ {ticker}: {exc}", style="bold red"))
        return None
    except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill a batch
        console.print(Text(f"✖ {ticker}: unexpected failure — {exc.__class__.__name__}: {exc}",
                           style="bold red"))
        return None

    narrative = None
    if use_llm:
        label = f"[magenta]{ticker}: local model synthesising (this can take a minute)..."
        if quiet:
            narrative = llm.synthesise(payload, numeric_authority=numeric_authority)
        else:
            with console.status(label):
                narrative = llm.synthesise(payload, numeric_authority=numeric_authority)
    else:
        engine_verdict = payload.get("verdict", {})
        engine_setup = dict(engine_verdict.get("trade_setup") or {})
        engine_setup.setdefault("author", "engine")
        narrative = {
            **llm._fallback(payload),
            "source": "deterministic",
            "llm_note": "LLM disabled via --no-llm",
            "numbers_source": "engine",
            "numbers_issues": [],
            "verdict": engine_verdict.get("verdict"),
            "conviction_pct": engine_verdict.get("conviction_pct"),
            "trade_setup": engine_setup,
            "engine_verdict": engine_verdict.get("verdict"),
            "engine_conviction_pct": engine_verdict.get("conviction_pct"),
            "engine_trade_setup": engine_setup,
        }

    payload["narrative"] = narrative
    return payload


@app.command()
def main(
    ticker: str = typer.Option(..., "--ticker", "-t",
                               help="Ticker symbol, or several separated by commas."),
    period: str = typer.Option("5y", "--period", "-p",
                               help="History window (1y, 2y, 3y, 5y, max)."),
    no_llm: bool = typer.Option(False, "--no-llm",
                                help="Skip the local model; use the deterministic narrative."),
    engine_numbers: bool = typer.Option(
        False, "--engine-numbers",
        help="Let the quant engine own the verdict and price levels; the model "
             "only writes the narrative. By default the model sets them, "
             "subject to arithmetic validation."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the on-disk cache."),
    no_options: bool = typer.Option(False, "--no-options",
                                    help="Skip option-chain analysis (faster)."),
    json_out: Path | None = typer.Option(None, "--json",
                                         help="Also write the full payload to this JSON file."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress spinners."),
    clear_cache: bool = typer.Option(False, "--clear-cache",
                                     help="Delete all cached data, then exit."),
) -> None:
    """Analyse one or more tickers and render a terminal dashboard."""
    if clear_cache:
        removed = cache_mod.clear()
        console.print(f"[green]Cleared {removed} cached entries.[/green]")
        raise typer.Exit(0)

    tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if not tickers:
        console.print("[red]No ticker supplied.[/red]")
        raise typer.Exit(2)

    if not no_llm:
        ok, detail = llm.available()
        if not ok:
            console.print(Text(f"ℹ Local model unavailable ({detail}). "
                               "Falling back to the deterministic narrative.", style="yellow"))
        else:
            console.print(Text(f"ℹ Using local model: {detail} — nothing leaves this machine.",
                               style="dim"))

    results: list[dict[str, Any]] = []
    for symbol in tickers:
        payload = _analyse_one(symbol, period, not no_cache, not no_llm, no_options,
                               quiet, numeric_authority=not engine_numbers)
        if payload is None:
            continue
        results.append(payload)
        report.render(payload, payload.get("narrative"))

    if json_out and results:
        try:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with json_out.open("w") as fh:
                json.dump(results if len(results) > 1 else results[0], fh,
                          indent=2, default=str)
            console.print(f"[green]Wrote payload to {json_out}[/green]")
        except OSError as exc:
            console.print(f"[red]Could not write {json_out}: {exc}[/red]")

    if not results:
        raise typer.Exit(1)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
