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

from analyzer import charts, store  # noqa: E402
from views.theme import css  # noqa: E402

# Appearance is stored, not just held in session, so the app opens in the mode
# you left it in.
_settings = store.load_settings()
if "appearance" not in st.session_state:
    st.session_state.appearance = _settings.get("appearance", "light")

st.markdown(css(st.session_state.appearance), unsafe_allow_html=True)
charts.set_mode(st.session_state.appearance)

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
    ]
)

navigation.run()

with st.sidebar:
    # On a shared host the session is the only store, so the user needs a way
    # to carry their watchlist and holdings between visits.
    if store.is_shared_host():
        with st.expander("Save / restore your data"):
            st.caption(
                "This is a shared server, so your holdings live in this browser "
                "session only — nothing about your portfolio is written to it. "
                "Download to keep them, upload to restore."
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

    st.markdown(
        "<div class='muted' style='margin-top:1.4rem;border-top:1px solid "
        "var(--glass-edge);padding-top:.9rem'>Analytical output only — not "
        "investment advice.<br>Market data from Yahoo Finance. Nothing else "
        "leaves this machine.</div>",
        unsafe_allow_html=True,
    )
