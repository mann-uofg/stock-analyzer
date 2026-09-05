"""Research view: full deep-dive on a single ticker."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from analyzer import (
    charts,
    horizon,
    interpret,
    llm,
    modelbench,
    patterns,
    sizing,
    store,
)
from analyzer.datafeed import DataError

from .common import (
    analyse,
    coerce_numeric,
    big,
    escape,
    finding,
    fmt,
    gauge_grid,
    logo,
    md_safe,
    meter,
    money,
    plain_summary,
    prices,
    quote,
    html,
    stat,
    stat_grid,
    synthesise,
    verdict_chip,
)


def _controls() -> dict[str, Any]:
    """The search row, at the top of the page.

    The ticker box and the Analyse button are the only things anyone touches
    every visit, so they sit in the row itself; the two sets of settings behind
    them are opened on the rare occasions they change.
    """
    search, action, opts, sizing = st.columns([5, 1.3, 1.25, 1.5],
                                              vertical_alignment="bottom")
    with search:
        symbol = st.text_input(
            "Symbol", value=st.session_state.get("ticker", ""),
            placeholder="AAPL, SHOP.TO, …",
        ).strip().upper()
    with action:
        run = st.button("Analyse", type="primary", width="stretch")

    with opts:
        with st.popover("Options", width="stretch"):
            period = st.select_slider(
                "History", options=["1y", "2y", "3y", "5y", "max"], value="5y"
            )
            include_options = st.toggle("Option chain", value=True)
            use_llm = st.toggle("Written analysis", value=True,
                                help="Adds a written read of the numbers.")
            authority = st.toggle(
                "Model sets the numbers", value=True, disabled=not use_llm,
                help="On: the model issues the verdict and price levels, checked "
                     "against the reward-to-risk floor. Off: the quant engine "
                     "sets them and the model only writes the commentary.",
            )
            ok, detail = llm.available()
            st.caption(detail if ok else f"Unavailable — {detail}")

            # The benchmark lives here as well as in scripts/ because once the
            # app is deployed the API key is in Streamlit secrets and there is
            # no shell to run a script from.
            if llm.provider() == "cloud":
                if st.button("Which model is best?", width="stretch",
                             help="Sends each candidate the real prompt and "
                                  "scores its JSON and arithmetic."):
                    st.session_state["_run_bench"] = True

    with sizing:
        with st.popover("Position sizing", width="stretch"):
            settings = store.load_settings()
            book_value = _portfolio_value()
            account = st.number_input(
                "Account size", min_value=0.0, step=100.0, format="%.2f",
                value=float(settings["account_value"] or book_value or 0.0),
                help="Leave at your imported portfolio's value, or set it "
                     "manually to include cash.",
            )
            risk = st.slider(
                "Risk per trade %", min_value=0.25, max_value=5.0, step=0.25,
                value=float(settings["risk_pct"]),
                help="What being stopped out costs you. 1-2% is the usual ceiling.",
            )
            cap = st.slider(
                "Max position %", min_value=5.0, max_value=100.0, step=5.0,
                value=float(settings["max_position_pct"]),
                help="Ceiling on any one holding, regardless of how tight the stop is.",
            )
            fractional = st.toggle("Fractional shares",
                                   value=bool(settings["allow_fractional"]))

            if (account, risk, cap, fractional) != (
                settings["account_value"], settings["risk_pct"],
                settings["max_position_pct"], settings["allow_fractional"],
            ):
                store.save_settings({
                    "account_value": account, "risk_pct": risk,
                    "max_position_pct": cap, "allow_fractional": fractional,
                })

    return {
        "symbol": symbol, "run": run, "period": period,
        "include_options": include_options, "use_llm": use_llm,
        "authority": authority, "account_value": account, "risk_pct": risk,
        "max_position_pct": cap, "allow_fractional": fractional,
    }


def _portfolio_value() -> float:
    """Current book value, used as the default account size."""
    total = 0.0
    for position in store.load_portfolio():
        price = quote(position["symbol"])
        if price and position.get("quantity"):
            total += price * position["quantity"]
    return round(total, 2)


def _header(payload: dict, call: dict, horizons: dict) -> None:
    quote_block = payload["quote"]
    profile = (payload.get("fundamental") or {}).get("profile") or {}
    tech = payload["technical"]

    left, right = st.columns([3, 1.2])
    with left:
        bits = [profile.get("name"), profile.get("sector"), profile.get("industry")]
        # Ticker is typed by the user and the profile fields come from Yahoo;
        # both are rendered through unsafe_allow_html, so both are escaped.
        html(
            f"<div class='hero-name'>{logo(payload['meta']['ticker'], 34)}"
            f"<h1>{escape(payload['meta']['ticker'])}</h1>"
            f"<span class='hero-price num'>{fmt(quote_block.get('spot'))}</span></div>"
            f"<div class='hero-sub'>"
            f"{' · '.join(escape(b) for b in bits if b)}</div>"
        )
    with right:
        html(verdict_chip(
            call["verdict"],
            f"{call['conviction_pct']:.0f}% conviction · by {call['author']}"
            if call.get("conviction_pct") is not None else "",
        ))

    rng = tech.get("range_52w") or {}
    performance = tech.get("performance_pct") or {}
    html(stat_grid([
        stat("Change", fmt(quote_block.get("change_pct"), 2, "%", signed=True),
             tone="up" if (quote_block.get("change_pct") or 0) >= 0 else "down"),
        stat("52-week range", f"{fmt(rng.get('low'), 0)}–{fmt(rng.get('high'), 0)}",
             fmt(rng.get("pct_from_high"), 1, "% from high", signed=True), small=True),
        stat("Market cap", big(quote_block.get("market_cap"))),
        stat("1 month", fmt(performance.get("1m"), 1, "%", signed=True)),
        stat("1 year", fmt(performance.get("1y"), 1, "%", signed=True)),
    ]))

    cols = st.columns([1, 1, 2])
    with cols[0]:
        html(meter(horizons.get("near_term_score"), "Near term"))
    with cols[1]:
        html(meter(horizons.get("long_term_score"), "Long term"))
    with cols[2]:
        if horizons.get("summary"):
            html(f"<div class='muted' style='padding-top:.15rem'>"
                   f"{horizons['summary']}</div>")


def _trade_setup(setup: dict, controls: dict) -> None:
    if not setup.get("valid"):
        st.info(f"No trade setup — {setup.get('reason', 'unavailable')}")
        return

    html(stat_grid([
        stat("Direction", setup["direction"].upper(),
             tone="up" if setup["direction"] == "long" else "down"),
        stat("Entry", f"{setup['entry_low']:,.2f} – {setup['entry_high']:,.2f}", small=True),
        stat("Stop", fmt(setup["stop_loss"]),
             f"−{setup.get('risk_pct')}% risk", tone="down"),
        stat("Target 1", fmt(setup["target_1"]), f"{setup['risk_reward_t1']}:1", tone="up"),
        stat("Target 2", fmt(setup["target_2"]), f"{setup['risk_reward_t2']}:1", tone="up"),
    ]))
    html(f"<div class='muted'>{setup.get('basis', '')}</div>")
    for adjustment in setup.get("adjustments") or []:
        html(finding("note", "Adjusted", adjustment))

    # --- How much to actually buy -----------------------------------------
    st.subheader("Position size")
    plan = sizing.sizing_for_setup(
        setup, controls["account_value"], controls["risk_pct"],
        controls["max_position_pct"], controls["allow_fractional"],
    )

    if not plan.get("valid"):
        for warning in plan.get("warnings", []):
            st.info(warning)
        return

    html(stat_grid([
        stat("Shares", f"{plan['shares']:,.4g}"),
        stat("Position", money(plan["position_value"]),
             f"{plan['position_pct']:.1f}% of account"),
        stat("At risk", money(plan["dollars_at_risk"]),
             f"{controls['risk_pct']:.2f}% of account", tone="down"),
        stat("Profit at T1", money(plan.get("profit_at_target_1")), tone="up"),
        stat("Profit at T2", money(plan.get("profit_at_target_2")), tone="up"),
    ]))
    html(
        f"<div class='muted'>Sized so a stop at {setup['stop_loss']:,.2f} costs "
        f"{money(plan['dollars_at_risk'])} — {controls['risk_pct']:.2f}% of a "
        f"{money(controls['account_value'])} account. Bound by the "
        f"{plan['bound_by']}; entry taken at {plan['entry_used']:,.2f}, the least "
        "favourable fill in the range.</div>"
    )
    for warning in plan.get("warnings", []):
        html(finding("warning", "Sizing caution", warning))


def render() -> None:
    controls = _controls()
    symbol = controls["symbol"]

    if symbol:
        st.session_state.ticker = symbol
    if controls["run"]:
        # Bound to the symbol it was requested for. A plain session flag would
        # mean that merely typing a different ticker - which reruns the script
        # on every keystroke commit - silently kicks off another multi-minute
        # local synthesis the user never asked for.
        st.session_state.llm_for = symbol

    if not symbol:
        st.markdown("# Research")
        st.markdown(
            "<div class='muted'>Enter a symbol above to analyse it. "
            "Use the exchange suffix for non-US listings — <code>SHOP.TO</code> "
            "for the TSX line, <code>SHOP</code> for the NYSE one.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner(f"Analysing {symbol}…"):
            payload = analyse(symbol, controls["period"], not controls["include_options"])
            price_df = prices(symbol, controls["period"])
    except DataError as exc:
        st.markdown(f"# {symbol}")
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.markdown(f"# {symbol}")
        st.error(f"Unexpected failure: {exc.__class__.__name__}: {exc}")
        return

    # Say where the write-up is actually coming from. Claiming "locally" while
    # the payload goes to ollama.com would misrepresent the one thing a reader
    # might care most about, and the timing differs by an order of magnitude.
    provider = llm.provider()
    spinner_text = {
        "cloud": "Writing the analysis in the cloud… usually under a minute.",
        "local": "Writing the analysis on this machine… one to three minutes.",
    }.get(provider, "Writing the analysis…")

    narrative = None
    if controls["use_llm"] and st.session_state.get("llm_for") == symbol:
        with st.spinner(spinner_text):
            narrative = synthesise(
                payload, controls["authority"],
                f"{symbol}:{controls['period']}:{controls['authority']}",
            )

    engine_verdict = payload.get("verdict", {}) or {}
    if narrative and narrative.get("numbers_source") == "llm":
        call = {
            "verdict": narrative.get("verdict"),
            "conviction_pct": narrative.get("conviction_pct"),
            "trade_setup": narrative.get("trade_setup", {}),
            "author": {"cloud": "cloud model", "local": "local model"}.get(
                provider, "model"),
        }
    else:
        call = {
            "verdict": engine_verdict.get("verdict"),
            "conviction_pct": engine_verdict.get("conviction_pct"),
            "trade_setup": engine_verdict.get("trade_setup", {}),
            "author": "quant engine",
        }

    horizons = horizon.compute(payload)
    _header(payload, call, horizons)

    if st.session_state.pop("_run_bench", False):
        _model_benchmark(payload)

    # The symbol has already been validated by a successful fetch here, so it
    # goes on the watchlist as typed.
    in_watchlist = symbol in store.watchlist_symbols()
    action = st.columns([1.6, 5])
    with action[0]:
        if in_watchlist:
            if st.button("Stop watching", width="stretch"):
                store.remove_from_watchlist(symbol)
                st.rerun()
        else:
            if st.button("Add to watchlist", width="stretch"):
                store.add_to_watchlist(symbol)
                st.rerun()

    # Engine notices use the same card as every other observation, so severity
    # reads consistently across the app rather than switching visual language.
    cards: list[str] = []
    if narrative and narrative.get("numbers_source") == "engine (model plan rejected)":
        cards.append(finding(
            "warning", "Model plan rejected",
            "It proposed its own levels but they failed validation, so the "
            "engine's are shown: "
            + "; ".join(narrative.get("numbers_issues") or []),
        ))
    for warning in engine_verdict.get("warnings") or []:
        headline, _, detail = warning.partition(" - ")
        cards.append(finding("warning", headline.strip(), detail.strip()))
    if cards:
        html("".join(cards))

    tabs = st.tabs(["Chart", "Patterns", "Analysis", "Technicals", "Options",
                    "Fundamentals", "Data"])

    with tabs[0]:
        controls_row = st.columns([5, 1])
        with controls_row[1]:
            months = st.select_slider("Window", options=[3, 6, 12, 24, 60], value=12,
                                      format_func=lambda m: f"{m}m")
            show_levels = st.toggle("Levels", value=True)
            show_plan = st.toggle("Trade plan", value=True)
        with controls_row[0]:
            st.plotly_chart(
                charts.price_chart(price_df, payload["technical"],
                                   call["trade_setup"] if show_plan else None,
                                   months, show_levels),
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )
        st.subheader("Trade setup")
        _trade_setup(call.get("trade_setup") or {}, controls)

    with tabs[1]:
        _patterns_tab(price_df)

    with tabs[2]:
        _analysis_tab(narrative, controls, engine_verdict, call)

    with tabs[3]:
        _technicals_tab(payload)

    with tabs[4]:
        _options_tab(payload.get("options") or {})

    with tabs[5]:
        _fundamentals_tab(payload.get("fundamental") or {})

    with tabs[6]:
        section = st.selectbox(
            "Section",
            ["verdict", "technical", "risk", "options", "fundamental", "quote", "meta", "news"],
        )
        st.json(payload.get(section), expanded=2)
        st.download_button(
            "Download analysis (JSON)",
            data=json.dumps({**payload, "narrative": narrative}, indent=2, default=str),
            file_name=f"{symbol}_analysis.json", mime="application/json",
        )


def _model_benchmark(payload: dict) -> None:
    """Score the cloud candidates on this ticker, live.

    Deliberately measured rather than argued: which model is best for this job
    depends on JSON reliability and arithmetic, not on general ability, and
    that is only knowable by running it.
    """
    st.header("Which cloud model should you use?")
    st.caption(
        f"Each candidate gets the same prompt the app sends for "
        f"{payload['meta']['ticker']}, and its answer goes through the same "
        "validator that guards your trade plan — with repair off, so this is "
        "what the model got right unaided. Runs serially; allow a few minutes."
    )

    table = st.empty()
    rows: list[dict] = []

    def _render() -> None:
        frame = pd.DataFrame([{
            "Model": r["model"],
            "Valid JSON": "yes" if r["valid_json"] else "no",
            "Fields": f"{r['fields']}/{len(modelbench.REQUIRED_FIELDS)}",
            "Arithmetic": "passed" if r["arithmetic"] else "failed",
            "R:R": r["risk_reward"],
            "Seconds": r["seconds"],
            "Note": r["note"][:60],
        } for r in rows])
        table.dataframe(frame, width="stretch", hide_index=True)

    progress = st.progress(0.0)
    for index, model in enumerate(modelbench.CANDIDATES, start=1):
        rows.append(modelbench.run_one(
            model, modelbench.SYSTEM_PROMPT_AUTHORITY,
            modelbench.build_user_prompt(payload, authority=True),
            payload.get("quote", {}).get("spot") or 0.0,
        ))
        _render()
        progress.progress(index / len(modelbench.CANDIDATES))
    progress.empty()

    verdict = modelbench.recommend(rows)
    if verdict["model"]:
        html(finding(
            "good", f"Use {verdict['model']}", verdict["reason"],
        ))
        st.code(f'OLLAMA_CLOUD_MODEL = "{verdict["model"]}"', language="toml")
        st.caption(
            "Put that in your .env locally, or in Streamlit secrets on the "
            "deployed app, then reload."
        )
    else:
        html(finding("warning", "No clear winner", verdict["reason"]))


def _patterns_tab(price_df) -> None:
    """Active chart formations, each with its measured record on this ticker."""
    read = patterns.analyse(price_df)

    bias = read.get("bias")
    tone = {"bullish": "good", "bearish": "critical", "mixed": "warning"}.get(bias, "note")
    html(finding(
        tone,
        f"Pattern read: {(bias or 'nothing active').upper()}",
        read.get("bias_detail") or "",
    ))

    if read["structural"]:
        st.header("Formations")
        for pattern in read["structural"]:
            arrow = {"bullish": "▲", "bearish": "▼"}.get(pattern["direction"], "◆")
            stage = pattern["stage"]
            html(stat_grid([
                stat("Pattern", f"{arrow} {pattern['name']}", small=True,
                     tone="up" if pattern["direction"] == "bullish"
                     else "down" if pattern["direction"] == "bearish" else ""),
                stat("Stage", stage.title(), sub="levels below are live"
                     if stage == "confirmed" else "not yet triggered"),
                stat("Trigger", fmt(pattern["trigger"])),
                stat("Target", fmt(pattern["target"]) if pattern["target"] else "—"),
                stat("Invalidated at", fmt(pattern["invalidation"]), tone="down"),
            ]))
            html(f"<div class='muted' style='margin-bottom:1.2rem'>{pattern['detail']}</div>")

    if read["candlesticks"]:
        st.header("Recent candles, and whether they have ever worked here")
        st.caption(
            "Hit rate is how often this stock was higher 20 sessions after the "
            "same signal. Base rate is how often it was higher over any 20 "
            "sessions. Only the gap between them is evidence — a 60% hit rate "
            "in a stock that rises 58% of the time regardless is noise."
        )
        rows = []
        for candle in read["candlesticks"]:
            edge = candle["edge"]
            horizon = edge["horizons"].get(20)
            if horizon:
                gap = horizon["edge"]
                verdict_txt = (
                    "no edge here" if abs(gap) < 5
                    else "worked historically" if (gap > 0) == (candle["direction"] == "bullish")
                    else "backfired historically"
                )
                rows.append({
                    "Pattern": candle["name"],
                    "Says": candle["direction"],
                    "Seen": edge["occurrences"],
                    "Hit rate %": horizon["hit_rate"],
                    "Base %": horizon["base_rate"],
                    "Edge pts": gap,
                    "Median %": horizon["median_return"],
                    "Verdict": verdict_txt,
                })
            else:
                rows.append({
                    "Pattern": candle["name"], "Says": candle["direction"],
                    "Seen": edge["occurrences"], "Hit rate %": None,
                    "Base %": None, "Edge pts": None, "Median %": None,
                    "Verdict": "too few occurrences to judge",
                })

        frame = coerce_numeric(pd.DataFrame(rows),
                               ["Seen", "Hit rate %", "Base %", "Edge pts", "Median %"])
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={
                "Hit rate %": st.column_config.NumberColumn(format="%.0f"),
                "Base %": st.column_config.NumberColumn(format="%.0f"),
                "Edge pts": st.column_config.NumberColumn(format="%+.0f"),
                "Median %": st.column_config.NumberColumn(format="%+.1f"),
                "Seen": st.column_config.NumberColumn(format="%d"),
            },
        )

        for candle in read["candlesticks"]:
            if candle["agrees_with_history"] is False:
                html(finding(
                    "warning",
                    f"{candle['name']} is supposed to be {candle['direction']} — "
                    "on this stock it has not been",
                    f"{candle['meaning']} Historically this signal preceded moves "
                    f"in the opposite direction here, so it is scored by what "
                    f"happened rather than by what it is meant to mean.",
                ))

    if not read["structural"] and not read["candlesticks"]:
        html(
            "<div class='muted'>No recognised formation is active. That is the "
            "normal state — a chart spends most of its life not forming anything "
            "worth naming, and forcing a pattern onto noise is how people lose "
            "money on this kind of analysis.</div>"
        )

    with st.expander("How to read this, and what it is worth"):
        st.markdown(
            "Chart patterns are the weakest evidence in this app. They are "
            "widely published, therefore widely traded, and their edge decays "
            "as a result. Published hit rates come from other decades and other "
            "instruments.\n\n"
            "That is why every candlestick here is re-measured **on this "
            "ticker's own history** rather than quoted from a book. Treat a "
            "pattern as a reason to look closer, never as a reason to trade on "
            "its own — and give real weight only to the ones whose edge over the "
            "base rate is large and based on a decent number of occurrences."
        )


def _analysis_tab(narrative, controls, engine_verdict, call) -> None:
    if not narrative:
        if controls["use_llm"]:
            st.info("Press **Analyse** above to generate the written analysis.")
        else:
            st.info("Enable **Written analysis** under Options, then press Analyse.")
        return

    st.caption(f"Source: {narrative.get('source')}"
               + (f" — {narrative['llm_note']}" if narrative.get("llm_note") else ""))

    if narrative.get("executive_summary"):
        st.write(md_safe(narrative["executive_summary"]))

    bull, bear = st.columns(2)
    with bull:
        st.subheader("Bull case")
        for item in narrative.get("bull_case") or []:
            st.success(md_safe(item))
    with bear:
        st.subheader("Bear case")
        for item in narrative.get("bear_case") or []:
            st.error(md_safe(item))

    if narrative.get("key_risk"):
        st.warning(f"**Key risk** — {md_safe(narrative['key_risk'])}")

    for title, key in (
        ("Technical", "technical_summary"),
        ("Risk and volatility", "risk_volatility_assessment"),
        ("Fundamentals and earnings", "fundamental_earnings_thesis"),
        ("Options positioning", "options_positioning"),
        ("Managing the trade", "trade_commentary"),
        ("How the model set its levels", "numbers_rationale"),
    ):
        if narrative.get(key):
            with st.expander(title):
                st.write(md_safe(narrative[key]))

    engine_setup = narrative.get("engine_trade_setup") or {}
    if narrative.get("numbers_source") == "llm" and engine_setup.get("valid"):
        setup = call.get("trade_setup") or {}
        st.subheader("Model vs engine")
        st.dataframe(
            pd.DataFrame([
                {"Metric": "Verdict", "Model": narrative.get("verdict"),
                 "Engine": narrative.get("engine_verdict")},
                {"Metric": "Conviction %", "Model": narrative.get("conviction_pct"),
                 "Engine": narrative.get("engine_conviction_pct")},
                *[{"Metric": label, "Model": setup.get(key), "Engine": engine_setup.get(key)}
                  for label, key in (("Entry low", "entry_low"), ("Entry high", "entry_high"),
                                     ("Stop", "stop_loss"), ("Target 1", "target_1"),
                                     ("Target 2", "target_2"), ("R:R", "risk_reward_t1"))],
            ]),
            width="stretch", hide_index=True,
        )


def _technicals_tab(payload: dict) -> None:
    tech, risk_panel = payload["technical"], payload["risk"]
    ma = tech.get("moving_averages", {})
    mom, vol, vlm = (tech.get("momentum", {}), tech.get("volatility", {}),
                     tech.get("volume", {}))

    readings = interpret.read_many([
        ("rsi_14", mom.get("rsi_14")),
        ("stoch_k", mom.get("stoch_k")),
        ("cci_20", mom.get("cci_20")),
        ("adx_14", vol.get("adx_14")),
        ("bb_percent_b", vol.get("bb_percent_b")),
        ("atr_percent", vol.get("atr_percent")),
        ("volume_ratio", vlm.get("volume_ratio")),
    ])
    html(plain_summary(interpret.summarise(readings, "the chart")))

    # Concerns first: someone scanning this page should meet the awkward
    # readings before the reassuring ones, not have to hunt for them.
    flagged = interpret.concerns(readings)
    if flagged:
        st.caption("Worth a closer look")
        html(gauge_grid(flagged))
        rest = [r for r in readings if r not in flagged]
        if rest:
            st.caption("Reading normally")
            html(gauge_grid(rest))
    else:
        html(gauge_grid(readings))

    with st.expander("The underlying numbers"):
        cols = st.columns(4)
        with cols[0]:
            st.markdown("**Trend**")
            st.metric("SMA 20", fmt(ma.get("sma_20")))
            st.metric("SMA 50", fmt(ma.get("sma_50")))
            st.metric("SMA 200", fmt(ma.get("sma_200")))
            cross = ma.get("golden_death_cross", {}) or {}
            st.caption(
                f"{ma.get('alignment') or '—'} · 50/200 {cross.get('state') or '—'}"
            )
        with cols[1]:
            st.markdown("**Momentum**")
            st.metric("RSI (14)", fmt(mom.get("rsi_14"), 1), mom.get("rsi_state"))
            st.metric("MACD", fmt(mom.get("macd")),
                      fmt(mom.get("macd_histogram"), 3, signed=True))
            st.metric("Stochastic %K", fmt(mom.get("stoch_k"), 1))
            st.metric("CCI (20)", fmt(mom.get("cci_20"), 1, signed=True))
        with cols[2]:
            st.markdown("**Volatility**")
            st.metric("ATR (14)", fmt(vol.get("atr_14")),
                      fmt(vol.get("atr_percent"), 2, "% of price"))
            st.metric("ADX (14)", fmt(vol.get("adx_14"), 1), vol.get("adx_state"))
            st.metric("Bollinger %B", fmt(vol.get("bb_percent_b"), 3))
            st.metric("Bandwidth", fmt(vol.get("bb_bandwidth"), 2))
        with cols[3]:
            st.markdown("**Volume**")
            st.metric("Last", big(vlm.get("last_volume")))
            st.metric("20-day average", big(vlm.get("avg_volume_20d")))
            st.metric("Ratio", fmt(vlm.get("volume_ratio"), 2, "x"),
                      "spike" if vlm.get("volume_spike") else None)
            st.metric("VWAP (14)", fmt(vlm.get("vwap_14")),
                      vlm.get("price_vs_vwap"))

    st.subheader("Market sensitivity")

    # Lead with the estimator Yahoo and Google publish, so the headline figure
    # is the one a reader can cross-check anywhere else.
    benchmarks = risk_panel.get("benchmarks") or {}
    headline = None
    for bench in ("SPY", "QQQ"):
        window = (benchmarks.get(bench) or {}).get("5y_monthly")
        if window and window.get("beta") is not None:
            headline = (bench, window)
            break
    if headline:
        bench_name, window = headline
        st.caption(f"Against {bench_name}, measured over five years of monthly moves")
        html(gauge_grid(interpret.read_many([
            ("beta", window.get("beta")),
            ("r_squared", window.get("r_squared")),
        ])))

    rows = []
    for bench, windows in (risk_panel.get("benchmarks") or {}).items():
        for label, vals in windows.items():
            rows.append({
                "Benchmark": bench,
                "Window": "5y monthly ★" if label == "5y_monthly" else label,
                "Beta": vals.get("beta"),
                "Jensen's alpha %": vals.get("alpha_annual_pct"),
                "R²": vals.get("r_squared"),
                "Weak": "⚠" if vals.get("low_explanatory_power") else "",
                "Obs": vals.get("observations") or vals.get("months"),
            })
    if rows:
        frame = pd.DataFrame(rows)
        for col in ("Beta", "Jensen's alpha %", "R²", "Obs"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={
                "Beta": st.column_config.NumberColumn(format="%.3f"),
                "Jensen's alpha %": st.column_config.NumberColumn(format="%.2f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
                "Obs": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "★ 5y monthly is the estimator Yahoo and Google publish — use that row "
            "to cross-check. The 1y/3y windows use daily returns and react faster "
            "to a change in regime. ⚠ marks R² below 0.10."
        )

    ratios, rvol = risk_panel.get("ratios", {}), risk_panel.get("volatility", {})

    st.subheader("Reward for the risk")
    html(gauge_grid(interpret.read_many([
        ("sharpe", ratios.get("sharpe")),
        ("sortino", ratios.get("sortino")),
        ("max_drawdown_1y_pct", risk_panel.get("max_drawdown_1y_pct")),
        ("hv_percentile_1y", rvol.get("hv_percentile_1y")),
    ])))

    cols = st.columns(2)
    cols[0].metric("Annualised volatility (30d)",
                   fmt(rvol.get("hv_30d_annual_pct"), 1, "%"),
                   help="How much it swings over a year, projected from the "
                        "last 30 days.")
    cols[1].metric("Annual return",
                   fmt(ratios.get("annual_return_pct"), 1, "%", signed=True),
                   help="Compound annual return over the analysed window.")


def _options_tab(opts: dict) -> None:
    if not opts.get("available"):
        st.info(f"Options unavailable — {opts.get('reason')}")
        return

    ivc, pcr = opts.get("iv_context", {}), opts.get("put_call_ratio", {})
    gamma = opts.get("gamma_exposure", {})
    readings = interpret.read_many([
        ("iv_hv_ratio", ivc.get("iv_hv_ratio")),
        ("put_call_ratio", pcr.get("open_interest")),
    ])
    if readings:
        html(plain_summary(interpret.summarise(readings, "the options market")))
        html(gauge_grid(readings))

    cols = st.columns(4)
    cols[0].metric("Expiry", opts["expiry"], f"{opts['days_to_expiry']} days")
    cols[1].metric("Expected swing (implied)", fmt(opts.get("atm_iv_pct"), 1, "%"),
                   help="The annual movement options traders are pricing in.")
    cols[2].metric("Actual swing (last 30d)", fmt(ivc.get("hv_30d_pct"), 1, "%"),
                   help="What the stock has really been doing, on the same "
                        "annualised scale — the comparison the gauge above makes.")
    cols[3].metric("P/C by volume", fmt(pcr.get("volume"), 2),
                   help="Today's put versus call trading. Noisier than the "
                        "open-interest version gauged above.")

    for note in (opts.get("data_quality") or {}).get("notes") or []:
        st.caption(note)

    smile = charts.iv_smile(opts)
    left, right = st.columns([3, 2])
    if smile:
        left.plotly_chart(smile, width="stretch",
                          config={"displayModeBar": False})
    with right:
        st.subheader("Gamma positioning")
        st.metric(f"Net near-the-money gamma ({gamma.get('weighted_by')})",
                  fmt(gamma.get("net_gamma"), 0, signed=True))
        st.metric("Peak call strike", fmt(gamma.get("peak_call_strike")))
        st.metric("Peak put strike", fmt(gamma.get("peak_put_strike")))
        if gamma.get("gamma_squeeze_flag"):
            st.success("Gamma squeeze setup detected")

    st.subheader("Near-the-money chain")
    side = st.radio("Side", ["calls", "puts"], horizontal=True, label_visibility="collapsed")
    chain = pd.DataFrame(opts.get("near_the_money", {}).get(side) or [])
    if chain.empty:
        st.info("No near-the-money contracts returned.")
    else:
        st.dataframe(
            chain[["strike", "mid", "iv_pct", "iv_source", "volume",
                   "open_interest", "delta", "gamma", "theta", "vega"]],
            width="stretch", hide_index=True,
            column_config={
                "iv_pct": st.column_config.NumberColumn("IV %", format="%.1f"),
                "iv_source": st.column_config.TextColumn("IV from"),
                "delta": st.column_config.NumberColumn(
                    format="%.3f",
                    help="How much the option moves per $1 move in the stock. "
                         "0.50 means it gains about 50c. Also a rough shorthand "
                         "for the chance it expires in the money."),
                "gamma": st.column_config.NumberColumn(
                    format="%.5f",
                    help="How fast delta itself changes. High gamma means the "
                         "option's behaviour shifts quickly as the stock moves."),
                "theta": st.column_config.NumberColumn(
                    format="%.3f",
                    help="What the option loses each day simply from time "
                         "passing. Always working against a buyer."),
                "vega": st.column_config.NumberColumn(
                    format="%.3f",
                    help="How much the option moves per 1 point change in "
                         "implied volatility — the fear premium."),
            },
        )
        st.caption("`solved` means implied volatility was recovered from the traded "
                   "price by inverting Black-Scholes, because the quoted value was "
                   "a placeholder.")


def _fundamentals_tab(fund: dict) -> None:
    val, earn = fund.get("valuation", {}), fund.get("earnings", {})
    cons = fund.get("consensus", {})

    readings = interpret.read_many([
        ("forward_pe", val.get("forward_pe")),
        ("trailing_pe", val.get("trailing_pe")),
        ("peg", val.get("peg_ratio")),
        ("ev_ebitda", val.get("ev_to_ebitda")),
        ("fcf_yield_pct", val.get("fcf_yield_pct")),
        ("days_to_earnings", earn.get("days_to_earnings")),
    ])
    html(plain_summary(interpret.summarise(readings, "the valuation")))
    html(gauge_grid(readings))

    st.caption(
        "Valuation bands are conventions, not verdicts — a fast-growing "
        "software company trades where a utility would look absurd. Compare "
        "these with companies in the same industry before drawing a conclusion."
    )

    with st.expander("Other fundamentals"):
        cols = st.columns(3)
        cols[0].metric("Price / sales", fmt(val.get("price_to_sales"), 1),
                       help="Market value against annual revenue. Useful when a "
                            "company has no profits yet.")
        cols[1].metric("Revenue YoY",
                       fmt(val.get("revenue_yoy_growth_pct"), 1, "%", signed=True),
                       help="Sales growth against the same quarter last year.")
        cols[2].metric("Profit margin", fmt(val.get("profit_margin_pct"), 1, "%"),
                       help="Profit per pound of sales. Higher usually means "
                            "more pricing power.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Earnings record")
        chart = charts.earnings_chart(earn)
        if chart:
            st.plotly_chart(chart, width="stretch",
                            config={"displayModeBar": False})
            st.caption("Bars are EPS surprise; the line is the next session's price "
                       "reaction. Beats with negative reactions mean the beat was "
                       "already priced in.")
        else:
            st.info("No earnings history for this symbol.")
    with right:
        st.subheader("Next catalyst")
        st.metric("Earnings date", earn.get("next_earnings_date") or "—",
                  f"{earn['days_to_earnings']} days away"
                  if earn.get("days_to_earnings") is not None else None)
        st.metric("Beat rate", fmt(earn.get("beat_rate_pct"), 0, "%"))
        st.metric("Average post-earnings move",
                  fmt(earn.get("avg_abs_post_earnings_move_pct"), 2, "%"))
        targets = cons.get("price_targets") or {}
        if targets.get("mean") and targets.get("current"):
            st.metric("Analyst mean target", fmt(targets["mean"]),
                      fmt((targets["mean"] / targets["current"] - 1) * 100, 1,
                          "% upside", signed=True))

    eps = cons.get("eps") or {}
    if eps:
        st.subheader("Forward consensus")
        st.dataframe(
            pd.DataFrame([
                {"Period": k.replace("_", " ").title(), "Consensus EPS": v.get("consensus"),
                 "Low": v.get("low"), "High": v.get("high"),
                 "YoY %": v.get("yoy_growth_pct"), "Analysts": v.get("analysts")}
                for k, v in eps.items()
            ]),
            width="stretch", hide_index=True,
        )
