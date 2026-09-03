"""The browser half of shared-host persistence.

On a shared host the server is the wrong place to keep holdings: the disk is
common to every visitor and is wiped whenever the container restarts. The
visitor's own browser has neither problem, so that is where their state lives.

This module owns the bridge to it. Everything about *what* is stored belongs to
``store``; all that happens here is mounting the frame and handing back what it
said.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

COMPONENT_NAME = "stock_analyzer_localstore"
_DIR = Path(__file__).parent / "localstore"

_component: Any = None


def _bridge() -> Any:
    """The declared component, built once per process."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component(COMPONENT_NAME, path=str(_DIR))
    return _component


def available() -> bool:
    """Whether a live Streamlit session exists to mount the frame into."""
    try:
        from streamlit.runtime import exists as runtime_exists

        return bool(runtime_exists())
    except Exception:
        return False


def sync(write: str | None = None, clear: bool = False) -> dict | None:
    """Mount the bridge, optionally writing, and return the browser's reply.

    ``None`` means the frame has not answered yet - which is not the same as
    the browser having nothing saved, and callers must not treat it as such.
    A dict with ``ready`` carries ``data``: the stored string, or None.
    """
    if not available():
        return None
    try:
        reply = _bridge()(
            write=write, clear=clear, default=None, key="_localstore_bridge"
        )
    except Exception:
        # A missing build directory or a blocked frame must not take the app
        # down; it just means this session will not persist.
        return None
    return reply if isinstance(reply, dict) else None
