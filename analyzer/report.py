"""Terminal rendering with rich.

Presentation rule: anything the data layer could not supply is rendered as a
dim "n/a" rather than a zero or a blank. On a trading desk a silent zero is
worse than an obvious gap.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

VERDICT_STYLE = {
    "STRONG BUY": "bold white on dark_green",
    "BUY": "bold green",
    "HOLD": "bold yellow",
    "SELL": "bold red",
    "STRONG SELL": "bold white on dark_red",
}

NA = Text("n/a", style="dim")


def _num(value: Any, digits: int = 2, suffix: str = "", signed: bool = False) -> Text:
    """Format a number, colouring by sign when meaningful."""
    if value is None or isinstance(value, bool):
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return Text(str(value))
    if number != number:  # NaN
        return NA
    text = f"{number:+,.{digits}f}" if signed else f"{number:,.{digits}f}"
    style = ""
    if signed:
        style = "green" if number > 0 else "red" if number < 0 else ""
    return Text(f"{text}{suffix}", style=style)


def _state(value: Any, good: tuple[str, ...] = (), bad: tuple[str, ...] = ()) -> Text:
    if value is None:
        return NA
    text = str(value)
    if text in good:
        return Text(text, style="green")
    if text in bad:
        return Text(text, style="red")
    return Text(text)


def _bar(pct: float | None, width: int = 24) -> Text:
    if pct is None:
        return NA
    filled = int(round(width * max(0.0, min(100.0, pct)) / 100))
    colour = "green" if pct >= 60 else "yellow" if pct >= 35 else "red"
    return Text("█" * filled, style=colour) + Text("░" * (width - filled), style="dim")


def _kv_table(title: str, rows: list[tuple[str, Any]]) -> Table:
    table = Table(
        title=title, box=box.SIMPLE_HEAD, title_style="bold cyan",
        title_justify="left", show_header=False, pad_edge=False, expand=True,
    )
    table.add_column("k", style="dim", no_wrap=True)
    table.add_column("v", justify="right")
    for key, value in rows:
        table.add_row(key, value if isinstance(value, Text) else Text(str(value)))
    return table


# --- Sections -------------------------------------------------------------


def effective_call(payload: dict, narrative: dict | None) -> dict:
    """Resolve whose verdict and levels are authoritative for this render.

    With numeric authority granted, a model plan that passed validation wins;
    otherwise the engine's numbers stand.
    """
    engine = payload.get("verdict", {}) or {}
    base = {
        "score_0_100": engine.get("score_0_100"),
        "warnings": engine.get("warnings", []),
    }
    if narrative and narrative.get("numbers_source") == "llm":
        return {
            **base,
            "verdict": narrative.get("verdict"),
            "conviction_pct": narrative.get("conviction_pct"),
            "trade_setup": narrative.get("trade_setup", {}),
            "author": "local model",
        }
    return {
        **base,
        "verdict": engine.get("verdict"),
        "conviction_pct": engine.get("conviction_pct"),
        "trade_setup": engine.get("trade_setup", {}),
        "author": "quant engine",
    }


def header(payload: dict, verdict: dict) -> Panel:
    meta, quote = payload["meta"], payload["quote"]
    profile = payload.get("fundamental", {}).get("profile", {})

    name = profile.get("name") or meta["ticker"]
    change = quote.get("change_pct")
    change_text = _num(change, 2, "%", signed=True)

    left = Text()
    left.append(f"{meta['ticker']}  ", style="bold white")
    left.append(f"{name}\n", style="dim")
    left.append(f"{quote.get('spot', 0):,.2f} ", style="bold")
    left.append(change_text)
    left.append(f"   {profile.get('sector') or ''}", style="dim")

    style = VERDICT_STYLE.get(verdict["verdict"], "bold")
    right = Group(
        Text(f" {verdict['verdict']} ", style=style),
        Text(f"Engine score {verdict['score_0_100']}/100", style="dim"),
        Text.assemble("Conviction ", _bar(verdict["conviction_pct"]),
                      f" {verdict['conviction_pct']:.0f}%"),
        Text(f"call by: {verdict['author']}", style="dim italic"),
    )

    return Panel(
        Columns([left, right], expand=True),
        box=box.ROUNDED,
        border_style="cyan",
        subtitle=f"[dim]{meta['generated_at']} · {meta['data_source']}[/dim]",
    )


def technical_tables(payload: dict) -> Columns:
    tech = payload["technical"]
    ma, mom = tech.get("moving_averages", {}), tech.get("momentum", {})
    vol, vlm = tech.get("volatility", {}), tech.get("volume", {})
    levels = tech.get("levels", {})

    cross = ma.get("golden_death_cross", {}) or {}
    cross_text = cross.get("state") or "n/a"
    if cross.get("event") and cross.get("bars_ago") is not None:
        cross_text = f"{cross['state']} ({cross['bars_ago']}d ago)"

    trend = _kv_table(
        "Trend & Moving Averages",
        [
            ("SMA 20 / 50 / 200", Text.assemble(
                _num(ma.get("sma_20")), " / ", _num(ma.get("sma_50")), " / ", _num(ma.get("sma_200")))),
            ("EMA 20 / 50 / 200", Text.assemble(
                _num(ma.get("ema_20")), " / ", _num(ma.get("ema_50")), " / ", _num(ma.get("ema_200")))),
            ("Price vs 50 / 200", Text.assemble(
                _state(ma.get("price_vs_sma50"), ("above",), ("below",)), " / ",
                _state(ma.get("price_vs_sma200"), ("above",), ("below",)))),
            ("MA alignment", _state(ma.get("alignment"), ("bullish_stacked",), ("bearish_stacked",))),
            ("50/200 cross", _state(cross_text, ("golden",), ("death",))),
            ("Support", Text(", ".join(f"{s:,.2f}" for s in levels.get("support", [])) or "n/a")),
            ("Resistance", Text(", ".join(f"{r:,.2f}" for r in levels.get("resistance", [])) or "n/a")),
        ],
    )

    momentum = _kv_table(
        "Momentum & Oscillators",
        [
            ("RSI (14)", Text.assemble(_num(mom.get("rsi_14"), 1), "  ",
                                       _state(mom.get("rsi_state"), ("oversold",), ("overbought",)))),
            ("MACD / signal", Text.assemble(_num(mom.get("macd"), 2), " / ", _num(mom.get("macd_signal"), 2))),
            ("MACD histogram", _num(mom.get("macd_histogram"), 3, signed=True)),
            ("Stochastic %K / %D", Text.assemble(_num(mom.get("stoch_k"), 1), " / ", _num(mom.get("stoch_d"), 1))),
            ("CCI (20)", _num(mom.get("cci_20"), 1, signed=True)),
            ("ADX (14)", Text.assemble(_num(vol.get("adx_14"), 1), "  ",
                                       _state(vol.get("adx_state"), ("strong_trend",), ("ranging",)))),
            ("+DI / -DI", Text.assemble(_num(vol.get("plus_di"), 1), " / ", _num(vol.get("minus_di"), 1))),
        ],
    )

    volatility = _kv_table(
        "Volatility & Channels",
        [
            ("Bollinger upper", _num(vol.get("bb_upper"))),
            ("Bollinger middle", _num(vol.get("bb_middle"))),
            ("Bollinger lower", _num(vol.get("bb_lower"))),
            ("%B / bandwidth", Text.assemble(_num(vol.get("bb_percent_b"), 3), " / ",
                                             _num(vol.get("bb_bandwidth"), 2))),
            ("ATR (14)", Text.assemble(_num(vol.get("atr_14")), " (",
                                       _num(vol.get("atr_percent"), 2, "%"), ")")),
        ],
    )

    volume = _kv_table(
        "Volume",
        [
            ("Last volume", _num(vlm.get("last_volume"), 0)),
            ("20-day average", _num(vlm.get("avg_volume_20d"), 0)),
            ("Ratio", Text.assemble(_num(vlm.get("volume_ratio"), 2, "x"),
                                    Text("  SPIKE", style="bold yellow") if vlm.get("volume_spike") else "")),
            ("OBV trend (20d)", _state(vlm.get("obv_trend_20d"), ("rising",), ("falling",))),
            ("VWAP (14)", _num(vlm.get("vwap_14"))),
            ("Price vs VWAP", _state(vlm.get("price_vs_vwap"), ("above",), ("below",))),
        ],
    )

    return Columns([trend, momentum, volatility, volume], expand=True, equal=True)


def risk_table(payload: dict) -> Table:
    risk_panel = payload["risk"]
    table = Table(
        title="Risk · Beta / Alpha / Volatility", box=box.SIMPLE_HEAD,
        title_style="bold cyan", title_justify="left", expand=True,
    )
    for col, justify in (("Benchmark", "left"), ("Window", "left"), ("Beta", "right"),
                         ("Jensen's Alpha", "right"), ("R²", "right"), ("Obs", "right")):
        table.add_column(col, justify=justify)

    weak_estimate = False
    for symbol, windows in (risk_panel.get("benchmarks") or {}).items():
        for label, vals in windows.items():
            r2 = vals.get("r_squared")
            weak = bool(vals.get("low_explanatory_power"))
            weak_estimate = weak_estimate or weak
            # The 5y monthly row is the public-convention estimator and carries
            # no alpha; mark it so it reads as the comparable figure.
            is_convention = label == "5y_monthly"
            table.add_row(
                symbol,
                Text("5y monthly *", style="bold cyan") if is_convention else Text(label),
                _num(vals.get("beta"), 3),
                Text("—", style="dim") if is_convention
                else _num(vals.get("alpha_annual_pct"), 2, "%", signed=True),
                Text.assemble(_num(r2, 3), Text(" ⚠", style="yellow") if weak else ""),
                Text(str(vals.get("observations") or vals.get("months") or "-"), style="dim"),
            )

    captions = ["* 5y monthly is the estimator Yahoo/Google publish — compare this one"]
    if weak_estimate:
        captions.append(
            "⚠ low R² — beta explains little of this name's variance over that "
            "window, so treat it as indicative only"
        )
    table.caption = "\n".join(captions)
    table.caption_style = "dim"

    vol = risk_panel.get("volatility", {})
    ratios = risk_panel.get("ratios", {})
    summary = _kv_table(
        "",
        [
            ("HV 30d (annualised)", _num(vol.get("hv_30d_annual_pct"), 2, "%")),
            ("HV percentile (1y)", _num(vol.get("hv_percentile_1y"), 0, "%")),
            ("Annual return / vol", Text.assemble(_num(ratios.get("annual_return_pct"), 1, "%", signed=True),
                                                  " / ", _num(ratios.get("annual_volatility_pct"), 1, "%"))),
            ("Sharpe / Sortino", Text.assemble(_num(ratios.get("sharpe"), 2), " / ", _num(ratios.get("sortino"), 2))),
            ("Max drawdown (1y)", _num(risk_panel.get("max_drawdown_1y_pct"), 2, "%")),
            ("Risk-free rate", _num(risk_panel.get("risk_free_rate_pct"), 2, "%")),
        ],
    )

    container = Table.grid(expand=True)
    container.add_column(ratio=3)
    container.add_column(ratio=2)
    container.add_row(table, summary)
    return container


def options_section(payload: dict) -> Group:
    opts = payload["options"]
    if not opts.get("available"):
        return Group(Panel(Text(f"Options unavailable — {opts.get('reason')}", style="yellow"),
                           title="Derivatives", border_style="yellow", box=box.ROUNDED))

    iv_ctx = opts.get("iv_context", {})
    pcr = opts.get("put_call_ratio", {})
    gamma = opts.get("gamma_exposure", {})

    summary = _kv_table(
        f"Derivatives · {opts['expiry']} ({opts['days_to_expiry']}d)",
        [
            ("ATM implied vol", _num(opts.get("atm_iv_pct"), 2, "%")),
            ("HV 30d", _num(iv_ctx.get("hv_30d_pct"), 2, "%")),
            ("IV / HV", _num(iv_ctx.get("iv_hv_ratio"), 3)),
            ("IV rank (proxy)", _num(iv_ctx.get("iv_rank_proxy_pct"), 1, "%")),
            ("IV percentile (proxy)", _num(iv_ctx.get("iv_percentile_proxy_pct"), 1, "%")),
            ("P/C by open interest", _num(pcr.get("open_interest"), 3)),
            ("P/C by volume", _num(pcr.get("volume"), 3)),
            (f"Net NTM gamma ({gamma.get('weighted_by')})", _num(gamma.get("net_gamma"), 0, signed=True)),
            ("Peak call / put strike", Text.assemble(_num(gamma.get("peak_call_strike")), " / ",
                                                     _num(gamma.get("peak_put_strike")))),
            ("Gamma squeeze flag", Text("YES", style="bold yellow") if gamma.get("gamma_squeeze_flag")
             else Text("no", style="dim")),
        ],
    )

    chain = Table(title="Near-the-money chain", box=box.SIMPLE_HEAD, title_style="bold cyan",
                  title_justify="left", expand=True)
    for col in ("Type", "Strike", "Mid", "IV", "Src", "Vol", "OI", "Delta", "Gamma", "Theta", "Vega"):
        chain.add_column(col, justify="right" if col != "Type" else "left")

    for kind, style in (("calls", "green"), ("puts", "red")):
        for row in (opts.get("near_the_money", {}).get(kind) or [])[:6]:
            chain.add_row(
                Text(kind[:-1].upper(), style=style),
                _num(row.get("strike")), _num(row.get("mid")),
                _num(row.get("iv_pct"), 1, "%"),
                Text(str(row.get("iv_source")), style="dim"),
                _num(row.get("volume"), 0), _num(row.get("open_interest"), 0),
                _num(row.get("delta"), 3), _num(row.get("gamma"), 5),
                _num(row.get("theta"), 3), _num(row.get("vega"), 3),
            )

    grid = Table.grid(expand=True)
    grid.add_column(ratio=2)
    grid.add_column(ratio=3)
    grid.add_row(summary, chain)

    parts: list[Any] = [grid]
    notes = (opts.get("data_quality") or {}).get("notes") or []
    if notes:
        parts.append(Panel(
            Text("\n".join(f"• {n}" for n in notes), style="yellow"),
            title="Data quality", border_style="yellow", box=box.ROUNDED,
        ))
    return Group(*parts)


def fundamentals_section(payload: dict) -> Columns:
    fund = payload["fundamental"]
    val, earn, cons = fund.get("valuation", {}), fund.get("earnings", {}), fund.get("consensus", {})

    valuation = _kv_table(
        "Valuation",
        [
            ("Trailing P/E", _num(val.get("trailing_pe"), 2)),
            ("Forward P/E", _num(val.get("forward_pe"), 2)),
            ("PEG", _num(val.get("peg_ratio"), 3)),
            ("Price/Sales", _num(val.get("price_to_sales"), 2)),
            ("EV/EBITDA", _num(val.get("ev_to_ebitda"), 2)),
            ("FCF yield", _num(val.get("fcf_yield_pct"), 2, "%")),
            ("Revenue YoY", _num(val.get("revenue_yoy_growth_pct"), 1, "%", signed=True)),
            ("Profit margin", _num(val.get("profit_margin_pct"), 1, "%")),
        ],
    )

    hist = Table(title="Earnings surprise history", box=box.SIMPLE_HEAD, title_style="bold cyan",
                 title_justify="left", expand=True)
    for col in ("Date", "Est EPS", "Actual", "Surprise"):
        hist.add_column(col, justify="right" if col != "Date" else "left")
    for row in (earn.get("history") or [])[:4]:
        hist.add_row(row["date"], _num(row.get("eps_estimate")), _num(row.get("eps_actual")),
                     _num(row.get("eps_surprise_pct"), 2, "%", signed=True))
    if not (earn.get("history") or []):
        hist.add_row(Text("no earnings history", style="dim"), NA, NA, NA)

    targets = cons.get("price_targets") or {}
    catalyst = _kv_table(
        "Catalyst & Consensus",
        [
            ("Next earnings", Text(str(earn.get("next_earnings_date") or "n/a"))),
            ("Days away", Text(str(earn.get("days_to_earnings") if earn.get("days_to_earnings") is not None else "n/a"),
                               style="bold yellow" if (earn.get("days_to_earnings") or 99) <= 7 else "")),
            ("Avg |post-earnings move|", _num(earn.get("avg_abs_post_earnings_move_pct"), 2, "%")),
            ("EPS beat rate", _num(earn.get("beat_rate_pct"), 0, "%")),
            ("Avg EPS surprise", _num(earn.get("avg_eps_surprise_pct"), 2, "%", signed=True)),
            ("Target mean / high", Text.assemble(_num(targets.get("mean")), " / ", _num(targets.get("high")))),
        ],
    )

    eps = cons.get("eps", {})
    fwd_rows: list[tuple[str, Any]] = []
    for label in ("current_quarter", "next_quarter", "current_year", "next_year"):
        block = eps.get(label)
        if not block:
            continue
        fwd_rows.append((
            label.replace("_", " ").title(),
            Text.assemble(_num(block.get("consensus"), 2), Text(
                f"  ({block['yoy_growth_pct']:+.0f}% YoY)" if block.get("yoy_growth_pct") is not None else "",
                style="dim")),
        ))
    forward = _kv_table("Forward EPS consensus", fwd_rows or [("no consensus data", NA)])

    return Columns([valuation, hist, catalyst, forward], expand=True, equal=True)


def divergence_panel(payload: dict, narrative: dict) -> Panel | None:
    """Show the engine's call beside the model's when they differ."""
    source = narrative.get("numbers_source")
    engine_setup = narrative.get("engine_trade_setup") or {}

    if source == "engine (model plan rejected)":
        body = Text("\n".join(f"• {i}" for i in narrative.get("numbers_issues", [])),
                    style="yellow")
        return Panel(
            Group(
                Text("The local model proposed its own trade plan, but it failed "
                     "arithmetic validation and was discarded. The engine's "
                     "numbers are shown above instead.\n"),
                body,
            ),
            title="⚠ Model plan rejected", border_style="yellow", box=box.ROUNDED,
        )

    if source != "llm" or not engine_setup.get("valid"):
        return None

    llm_setup = narrative.get("trade_setup", {}) or {}
    same_verdict = narrative.get("verdict") == narrative.get("engine_verdict")
    keys = ("entry_low", "entry_high", "stop_loss", "target_1", "target_2",
            "risk_reward_t1")
    if same_verdict and all(
        llm_setup.get(k) == engine_setup.get(k) for k in keys
    ):
        return None

    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("", style="dim")
    table.add_column("Local model", justify="right", style="magenta")
    table.add_column("Quant engine", justify="right", style="cyan")

    table.add_row("Verdict", Text(str(narrative.get("verdict"))),
                  Text(str(narrative.get("engine_verdict"))))
    table.add_row("Conviction", _num(narrative.get("conviction_pct"), 1, "%"),
                  _num(narrative.get("engine_conviction_pct"), 1, "%"))
    for label, key in (("Entry low", "entry_low"), ("Entry high", "entry_high"),
                       ("Stop loss", "stop_loss"), ("Target 1", "target_1"),
                       ("Target 2", "target_2"), ("R:R target 1", "risk_reward_t1")):
        table.add_row(label, _num(llm_setup.get(key)), _num(engine_setup.get(key)))

    return Panel(table, title="Model vs engine — where they differ",
                 border_style="magenta", box=box.ROUNDED)


def trade_panel(payload: dict, verdict: dict) -> Panel:
    setup = verdict.get("trade_setup", {}) or {}

    if not setup.get("valid"):
        return Panel(Text(f"No trade setup: {setup.get('reason')}", style="yellow"),
                     title="Trade Setup", border_style="yellow", box=box.ROUNDED)

    table = Table.grid(padding=(0, 3))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Direction", Text(setup["direction"].upper(),
                                    style="bold green" if setup["direction"] == "long" else "bold red"))
    table.add_row("Entry range", Text(f"{setup['entry_low']:,.2f} — {setup['entry_high']:,.2f}", style="bold"))
    table.add_row("Stop loss", Text(f"{setup['stop_loss']:,.2f}   (risk {setup['risk_pct']}% / "
                                    f"{setup['risk_per_share']:,.2f} per share)", style="red"))
    t1 = f"{setup['target_1']:,.2f}   R:R {setup['risk_reward_t1']}:1"
    if setup.get("target_1_synthetic"):
        t1 += "  [synthetic — no structural level satisfied the R:R floor]"
    table.add_row("Target 1", Text(t1, style="green"))
    table.add_row("Target 2", Text(f"{setup['target_2']:,.2f}   R:R {setup['risk_reward_t2']}:1", style="green"))
    if setup.get("atr_used"):
        table.add_row("ATR used", Text(f"{setup['atr_used']:,.2f}"))
    table.add_row("Basis", Text(setup["basis"], style="dim"))

    parts: list[Any] = [table]
    for adjustment in setup.get("adjustments") or []:
        parts.append(Text(f"✎  {adjustment}", style="cyan"))
    if verdict.get("warnings"):
        parts.append(Text())
        for warning in verdict["warnings"]:
            parts.append(Text(f"⚠  {warning}", style="yellow"))

    author = "set by the local model" if setup.get("author") == "llm" else "computed by the quant engine"
    return Panel(Group(*parts), title="Trade Setup & Risk Management",
                 subtitle=f"[dim]levels {author}[/dim]",
                 border_style="green", box=box.ROUNDED)


def signals_panel(payload: dict) -> Panel:
    buckets = payload["verdict"]["buckets"]
    table = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True)
    table.add_column("Bucket", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Contribution", justify="right")
    table.add_column("Drivers")

    for name, bucket in buckets.items():
        table.add_row(
            name.title(),
            _num(bucket["score"], 3, signed=True),
            _num(bucket["weight"], 2),
            _num(bucket["contribution"], 4, signed=True),
            Text("\n".join(f"• {r}" for r in bucket["reasons"]), style="dim"),
        )
    return Panel(table, title="Key Signals Matrix", border_style="cyan", box=box.ROUNDED)


def narrative_panels(narrative: dict) -> Group:
    source = narrative.get("source", "unknown")
    note = narrative.get("llm_note")
    subtitle = f"[dim]source: {source}" + (f" — {note}" if note else "") + "[/dim]"

    sections = [
        ("Executive Summary", narrative.get("executive_summary")),
        ("Technical", narrative.get("technical_summary")),
        ("Risk & Volatility", narrative.get("risk_volatility_assessment")),
        ("Fundamental & Earnings", narrative.get("fundamental_earnings_thesis")),
        ("Options Positioning", narrative.get("options_positioning")),
        ("Trade Commentary", narrative.get("trade_commentary")),
    ]
    body = []
    for title, text in sections:
        if not text:
            continue
        body.append(Text(title, style="bold cyan"))
        body.append(Text(text))
        body.append(Text())

    bull = Panel(Text("\n\n".join(f"▲ {b}" for b in narrative.get("bull_case", [])) or "n/a"),
                 title="Bull Case", border_style="green", box=box.ROUNDED)
    bear = Panel(Text("\n\n".join(f"▼ {b}" for b in narrative.get("bear_case", [])) or "n/a"),
                 title="Bear Case", border_style="red", box=box.ROUNDED)

    # A grid with equal ratios splits the width evenly; Columns sizes to
    # content and leaves the two cases visibly lopsided.
    cases = Table.grid(expand=True, padding=(0, 1))
    cases.add_column(ratio=1)
    cases.add_column(ratio=1)
    cases.add_row(bull, bear)

    parts: list[Any] = [Panel(Group(*body), title="Analyst Synthesis", subtitle=subtitle,
                              border_style="magenta", box=box.ROUNDED),
                        cases]
    if narrative.get("key_risk"):
        parts.append(Panel(Text(narrative["key_risk"], style="yellow"),
                           title="Key Risk", border_style="yellow", box=box.ROUNDED))
    return Group(*parts)


def render(payload: dict, narrative: dict | None = None, sections: set[str] | None = None) -> None:
    """Render the full dashboard to the terminal."""
    show = sections or {"header", "technical", "risk", "options", "fundamentals",
                        "signals", "trade", "narrative"}
    call = effective_call(payload, narrative)

    console.print()
    if "header" in show:
        console.print(header(payload, call))
    if "technical" in show:
        console.print(technical_tables(payload))
    if "risk" in show:
        console.print(risk_table(payload))
    if "options" in show:
        console.print(options_section(payload))
    if "fundamentals" in show:
        console.print(fundamentals_section(payload))
    if "signals" in show:
        console.print(signals_panel(payload))
    if "trade" in show:
        console.print(trade_panel(payload, call))
        if narrative:
            divergence = divergence_panel(payload, narrative)
            if divergence:
                console.print(divergence)
    if "narrative" in show and narrative:
        console.print(narrative_panels(narrative))
    console.print(
        Text("Analytical output only — not investment advice. Verify before trading.",
             style="dim italic"), justify="center",
    )
    console.print()
