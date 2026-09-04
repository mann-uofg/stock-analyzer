"""Persistence for the watchlist, portfolio and settings.

Two backends, chosen automatically:

* **Local** - plain JSON under ``data/``, written atomically. No database, no
  account: the file sits next to the code and is trivially inspectable or
  deletable.
* **Shared host** - the visitor's own browser, cached in the session.

That second mode is not a nicety. On a deployed instance the filesystem is
shared by every visitor, so writing holdings to ``data/portfolio.json`` would
publish one person's positions to everyone who opened the URL, and an ephemeral
container would lose them on the next restart regardless.

Session state alone solved the leak but not the loss: it evaporates the moment
the tab closes, so every visit started blank. State is therefore mirrored into
the browser's own storage, which survives closing the tab, the app sleeping and
redeploys, while still never reaching the server's disk. There is no account
and no password, because there is nothing on a server to protect - the data
never leaves the machine it was entered on.

The cost of that choice is honest and worth stating: it is per-browser. Another
device starts empty, and clearing site data clears this too, which is what the
JSON export is for.

Every read tolerates a missing or corrupt file by returning the empty default,
so a bad write can never brick the app.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import browserstore
from .config import ROOT


def _session_state() -> Any | None:
    """Streamlit's per-session store, when running inside a live session."""
    try:
        import streamlit as st

        from streamlit.runtime import exists as runtime_exists
    except Exception:
        return None
    try:
        return st.session_state if runtime_exists() else None
    except Exception:
        return None


def is_shared_host() -> bool:
    """Whether this process serves more than one person.

    Streamlit Community Cloud mounts the repository at /mount/src; the
    environment variable is the manual override for any other host.
    """
    if os.getenv("STOCK_ANALYZER_SHARED", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return Path("/mount/src").exists()

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    # 0 means "use the imported portfolio's market value".
    "account_value": 0.0,
    "risk_pct": 1.5,
    "max_position_pct": 20.0,
    "allow_fractional": True,
    "appearance": "light",
}

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(path: Path, default: Any) -> Any:
    # On a shared host the session is the only correct place to look: another
    # visitor's holdings must never be served as this visitor's.
    if is_shared_host():
        session = _session_state()
        if session is not None:
            stored = session.get(f"_store_{path.stem}")
            return default if stored is None else stored
        return default

    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default
    if not isinstance(payload, dict):
        return default
    return payload.get("data", default)


def _write(path: Path, data: Any) -> None:
    """Atomic write: temp file in the same directory, then rename."""
    if is_shared_host():
        session = _session_state()
        if session is not None:
            session[f"_store_{path.stem}"] = data
            session[f"_store_{path.stem}_updated"] = _now()
            # The browser is written on the next script run rather than here:
            # a component can only mount at one point in the layout, and every
            # mutation in this app is followed by a rerun.
            session[_DIRTY] = True
        return

    payload = {"schema": SCHEMA_VERSION, "updated": _now(), "data": data}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


STATE_KEYS = ("watchlist", "portfolio", "settings")
_DIRTY = "_localstore_dirty"
_HYDRATED = "_localstore_hydrated"


def _current_blob() -> str:
    """Everything this session holds, as one string for the browser."""
    session = _session_state()
    data = {key: (session.get(f"_store_{key}") if session else None)
            for key in STATE_KEYS}
    return json.dumps({"schema": SCHEMA_VERSION, "updated": _now(), "data": data})


def _apply_blob(raw: str) -> None:
    """Load a stored blob into the session, ignoring anything malformed."""
    session = _session_state()
    if session is None:
        return
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(payload, dict):
        return
    stored = payload.get("data")
    if not isinstance(stored, dict):
        return
    for key in STATE_KEYS:
        value = stored.get(key)
        if value is not None:
            session[f"_store_{key}"] = value


def sync_browser() -> None:
    """Mirror this session against the browser's own storage.

    Called once per script run, before anything reads state. The first run
    only mounts the frame - the browser's answer arrives on the run after, so
    hydration deliberately waits for an explicit ``ready`` rather than treating
    silence as "nothing saved", which would blank real holdings on first paint.
    """
    if not is_shared_host():
        return
    session = _session_state()
    if session is None:
        return

    hydrated = bool(session.get(_HYDRATED))
    # Only push once there is something of this session's own to push; writing
    # before hydration would overwrite saved data with an empty session.
    write = _current_blob() if (hydrated and session.get(_DIRTY)) else None

    reply = browserstore.sync(write=write)

    if write is not None:
        session[_DIRTY] = False

    if not hydrated and isinstance(reply, dict) and reply.get("ready"):
        raw = reply.get("data")
        if isinstance(raw, str) and raw:
            _apply_blob(raw)
        session[_HYDRATED] = True


def forget_browser() -> None:
    """Erase this browser's saved copy and the session with it."""
    session = _session_state()
    if session is None:
        return
    browserstore.sync(clear=True)
    for key in STATE_KEYS:
        session.pop(f"_store_{key}", None)
        session.pop(f"_store_{key}_updated", None)
    session[_DIRTY] = False


def saved_in_browser() -> bool:
    """Whether this session has been restored from, or saved to, the browser."""
    session = _session_state()
    return bool(session and session.get(_HYDRATED))


def export_state() -> dict[str, Any]:
    """Everything the user has entered, for download.

    On a shared host the session evaporates when the tab closes, so this plus
    ``import_state`` is how a portfolio survives - a small file the user keeps,
    rather than positions sitting on someone else's server.
    """
    return {
        "schema": SCHEMA_VERSION,
        "exported": _now(),
        "watchlist": load_watchlist(),
        "portfolio": load_portfolio(),
        "settings": load_settings(),
    }


def import_state(payload: dict[str, Any]) -> list[str]:
    """Restore an exported bundle. Returns a note of what was loaded."""
    notes: list[str] = []
    if not isinstance(payload, dict):
        return ["That file was not a saved state bundle."]

    watchlist = payload.get("watchlist")
    if isinstance(watchlist, list):
        save_watchlist(watchlist)
        notes.append(f"{len(watchlist)} watchlist symbols")

    portfolio = payload.get("portfolio")
    if isinstance(portfolio, list):
        save_portfolio(portfolio)
        notes.append(f"{len(portfolio)} positions")

    settings = payload.get("settings")
    if isinstance(settings, dict):
        save_settings({**load_settings(), **settings})
        notes.append("settings")

    return notes or ["Nothing recognisable in that file."]


# --- Watchlist ------------------------------------------------------------


def load_watchlist() -> list[dict[str, Any]]:
    """Watchlist entries: ``{symbol, note, added}``."""
    entries = _read(WATCHLIST_FILE, [])
    return entries if isinstance(entries, list) else []


def save_watchlist(entries: list[dict[str, Any]]) -> None:
    _write(WATCHLIST_FILE, entries)


def add_to_watchlist(symbol: str, note: str = "") -> tuple[bool, str]:
    """Add a symbol. Returns ``(changed, message)``."""
    symbol = symbol.strip().upper()
    if not symbol:
        return False, "Enter a symbol."

    entries = load_watchlist()
    if any(e.get("symbol") == symbol for e in entries):
        return False, f"{symbol} is already on the watchlist."

    entries.append({"symbol": symbol, "note": note.strip(), "added": _now()})
    save_watchlist(entries)
    return True, f"Added {symbol}."


def add_many_to_watchlist(symbols: list[str],
                          note: str = "") -> tuple[list[str], list[str]]:
    """Add several symbols in a single write. Returns ``(added, skipped)``.

    One write, not one per symbol: each write marks the session dirty and every
    caller reruns afterwards, so adding ten names one at a time meant ten
    reruns and - before screening was cached per symbol - ten full rescreens.
    """
    entries = load_watchlist()
    known = {e.get("symbol") for e in entries}
    added: list[str] = []
    skipped: list[str] = []

    for raw in symbols:
        symbol = raw.strip().upper()
        if not symbol:
            continue
        if symbol in known:
            skipped.append(symbol)
            continue
        entries.append({"symbol": symbol, "note": note.strip(), "added": _now()})
        known.add(symbol)
        added.append(symbol)

    if added:
        save_watchlist(entries)
    return added, skipped


def remove_from_watchlist(symbol: str) -> None:
    symbol = symbol.strip().upper()
    save_watchlist([e for e in load_watchlist() if e.get("symbol") != symbol])


def watchlist_symbols() -> list[str]:
    return [e["symbol"] for e in load_watchlist() if e.get("symbol")]


# --- Portfolio ------------------------------------------------------------


def load_portfolio() -> list[dict[str, Any]]:
    """Positions: ``{symbol, quantity, avg_cost, account}``."""
    positions = _read(PORTFOLIO_FILE, [])
    return positions if isinstance(positions, list) else []


def save_portfolio(positions: list[dict[str, Any]]) -> None:
    _write(PORTFOLIO_FILE, positions)


def clear_portfolio() -> None:
    save_portfolio([])


def load_settings() -> dict[str, Any]:
    """User settings, with defaults filled in for anything missing."""
    stored = _read(SETTINGS_FILE, {})
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(stored, dict):
        settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    _write(SETTINGS_FILE, {k: v for k, v in settings.items() if k in DEFAULT_SETTINGS})


def last_updated(path: Path) -> str | None:
    """ISO timestamp of the last write, or None."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh).get("updated")
    except (json.JSONDecodeError, OSError):
        return None
