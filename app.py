#!/usr/bin/env python3
"""Stock analyzer — local web interface.

    streamlit run app.py

Runs entirely on your machine. Telemetry is off and the server binds to
loopback (see ``.streamlit/config.toml``); the only outbound traffic is
yfinance fetching market data.
"""

from __future__ import annotations

import json
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analyzer import charts, llm, store  # noqa: E402
from views.theme import css  # noqa: E402

# An escape hatch, reachable as ?reset=1.
#
# Persisted state is restored on every load, so a saved watchlist that cannot
# be analysed re-triggers the same failure on each visit and refreshing never
# clears it. Without a way in through the URL the only remedy would be the
# browser's own developer tools, which is not a remedy at all.
_reset = st.query_params.get("reset") == "1"
if _reset:
    store.forget_browser()
    # Clearing the query parameter reruns the script, so the confirmation has
    # to survive in session state or it is never rendered.
    st.session_state["_was_reset"] = True
else:
    # Restore this browser's saved state before anything reads it. On a shared
    # host the session starts empty on every visit, so without this the first
    # read below would see nothing and the app would open blank each time.
    store.sync_browser()

# Appearance is stored, not just held in session, so the app opens in the mode
# you left it in.
_settings = store.load_settings()
if "appearance" not in st.session_state:
    st.session_state.appearance = _settings.get("appearance", "light")

st.markdown(css(st.session_state.appearance), unsafe_allow_html=True)
charts.set_mode(st.session_state.appearance)

from views import earnings as earnings_view  # noqa: E402
from views import news as news_view  # noqa: E402
from views import portfolio as portfolio_view  # noqa: E402
from views import research as research_view  # noqa: E402
from views import watchlist as watchlist_view  # noqa: E402

navigation = st.navigation(
    [
        # The default page always serves from "/" - giving it its own url_path
        # as well makes that path a 404, which greets the user with a
        # "page not found" dialog on an otherwise valid-looking URL.
        st.Page(research_view.render, title="Research", default=True),
        st.Page(watchlist_view.render, title="Watchlist", url_path="watchlist"),
        st.Page(portfolio_view.render, title="Portfolio", url_path="portfolio"),
        st.Page(news_view.render, title="News", url_path="news"),
        st.Page(earnings_view.render, title="Earnings", url_path="earnings"),
    ]
)

if _reset:
    st.query_params.clear()

if st.session_state.pop("_was_reset", False):
    st.warning(
        "Saved data cleared for this browser. Everything starts fresh — "
        "re-import your portfolio, or restore the JSON export from the "
        "sidebar if you have one."
    )

navigation.run()

with st.sidebar:
    # Holdings live in this browser. The export exists for the two cases that
    # storage cannot cover: moving to another device, and clearing site data.
    if store.is_shared_host():
        with st.expander("Your data"):
            st.caption(
                "Your holdings, watchlist and settings save automatically in "
                "this browser and are restored when you come back — no account, "
                "and nothing about your portfolio is stored on the server. "
                "Because it is tied to this browser, download a copy to move to "
                "another device or before clearing site data."
            )
            st.download_button(
                "Download my data",
                data=json.dumps(store.export_state(), indent=2, default=str),
                file_name="stock-analyzer-data.json",
                mime="application/json",
                width="stretch",
            )
            restore = st.file_uploader("Restore", type=["json"],
                                       label_visibility="collapsed")
            if restore is not None:
                try:
                    loaded = store.import_state(json.load(restore))
                    st.success("Restored " + ", ".join(loaded))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read that file: {exc}")

            # Erasing is irreversible and there is no server-side copy to
            # recover from, so it takes a deliberate second action.
            st.divider()
            confirm = st.checkbox("I want to erase my saved data")
            if st.button("Forget this browser", width="stretch",
                         disabled=not confirm):
                store.forget_browser()
                st.success("Erased. Nothing of yours is left in this browser.")
                st.rerun()

    st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
    modes = {"light": "Light", "dark": "Dark"}
    chosen = st.segmented_control(
        "Appearance", list(modes), format_func=lambda m: modes[m],
        default=st.session_state.appearance, key="appearance_picker",
    )
    if chosen and chosen != st.session_state.appearance:
        st.session_state.appearance = chosen
        store.save_settings({**store.load_settings(), "appearance": chosen})
        st.rerun()

    # Where the data goes depends on which model is answering, so the footer
    # reads the live provider rather than asserting a privacy guarantee that
    # stopped being true the moment this moved to a cloud model.
    where = {
        "cloud": "Market data from Yahoo Finance. Written analysis is generated "
                 "by a cloud model, which receives the figures for the ticker "
                 "you analyse.",
        "local": "Market data from Yahoo Finance. Nothing else leaves this "
                 "machine.",
    }.get(llm.provider(), "Market data from Yahoo Finance.")

    st.markdown(
        "<div class='muted' style='margin-top:1.4rem;border-top:1px solid "
        "var(--glass-edge);padding-top:.9rem'>Analytical output only — not "
        f"investment advice.<br>{where}</div>",
        unsafe_allow_html=True,
    )
