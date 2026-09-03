"""Tests for mirroring shared-host state into the visitor's browser.

The dangerous case is not a failed save, it is a premature one. The bridge
answers on the script run *after* it mounts, so for one run the session looks
empty while the browser actually holds a full portfolio. Writing during that
window, or concluding "nothing is saved", would destroy real data - so most of
what is checked here is that the code refuses to act until the browser has
explicitly said it is ready.
"""

from __future__ import annotations

import json

import pytest

from analyzer import store


@pytest.fixture
def session(monkeypatch):
    """A stand-in for st.session_state on a shared host."""
    state: dict = {}
    monkeypatch.setattr(store, "is_shared_host", lambda: True)
    monkeypatch.setattr(store, "_session_state", lambda: state)
    return state


def _replies(monkeypatch, *answers):
    """Queue the bridge's replies, recording what was written."""
    calls: list = []
    queue = list(answers)

    def fake_sync(write=None, clear=False):
        calls.append({"write": write, "clear": clear})
        return queue.pop(0) if queue else None

    monkeypatch.setattr(store.browserstore, "sync", fake_sync)
    return calls


def _blob(**data) -> str:
    return json.dumps({"schema": 1, "updated": "now", "data": data})


class TestHydration:
    def test_first_run_does_not_conclude_anything(self, session, monkeypatch):
        # The frame has not answered. Treating that as "nothing saved" is the
        # bug that would blank a real portfolio on every visit.
        _replies(monkeypatch, None)
        store.sync_browser()
        assert not session.get(store._HYDRATED)

    def test_second_run_loads_what_the_browser_held(self, session, monkeypatch):
        _replies(monkeypatch, None,
                 {"ready": True, "data": _blob(portfolio=[{"symbol": "NVDA"}])})
        store.sync_browser()
        store.sync_browser()
        assert session["_store_portfolio"] == [{"symbol": "NVDA"}]
        assert session[store._HYDRATED] is True

    def test_ready_with_nothing_saved_still_completes(self, session, monkeypatch):
        _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        assert session[store._HYDRATED] is True
        assert "_store_portfolio" not in session

    def test_hydration_happens_once(self, session, monkeypatch):
        _replies(monkeypatch,
                 {"ready": True, "data": _blob(watchlist=[{"symbol": "AMD"}])},
                 {"ready": True, "data": _blob(watchlist=[{"symbol": "GONE"}])})
        store.sync_browser()
        store.sync_browser()
        # A later reply must not silently replace live session state.
        assert session["_store_watchlist"] == [{"symbol": "AMD"}]


class TestWriting:
    def test_nothing_is_written_before_hydration(self, session, monkeypatch):
        # Pushing an empty session before the browser has answered would
        # overwrite saved holdings with nothing.
        session[store._DIRTY] = True
        calls = _replies(monkeypatch, None)
        store.sync_browser()
        assert calls[0]["write"] is None

    def test_dirty_state_is_pushed_once_hydrated(self, session, monkeypatch):
        session[store._HYDRATED] = True
        session[store._DIRTY] = True
        session["_store_portfolio"] = [{"symbol": "SHOP.TO"}]
        calls = _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        assert "SHOP.TO" in calls[0]["write"]

    def test_dirty_clears_after_a_push(self, session, monkeypatch):
        session[store._HYDRATED] = True
        session[store._DIRTY] = True
        _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        assert session[store._DIRTY] is False

    def test_clean_state_is_not_rewritten(self, session, monkeypatch):
        session[store._HYDRATED] = True
        calls = _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        assert calls[0]["write"] is None

    def test_saving_marks_the_session_dirty(self, session, monkeypatch):
        store.save_portfolio([{"symbol": "NVDA", "quantity": 3}])
        assert session[store._DIRTY] is True

    def test_a_save_survives_the_round_trip(self, session, monkeypatch):
        session[store._HYDRATED] = True
        store.save_portfolio([{"symbol": "NVDA", "quantity": 3}])
        calls = _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        written = json.loads(calls[0]["write"])
        assert written["data"]["portfolio"][0]["symbol"] == "NVDA"


class TestCorruptAndHostile:
    @pytest.mark.parametrize("raw", [
        "not json at all", "[]", "null", '{"data": "not a dict"}', '{"no_data": 1}',
    ])
    def test_unreadable_storage_never_raises(self, session, monkeypatch, raw):
        _replies(monkeypatch, {"ready": True, "data": raw})
        store.sync_browser()  # must not raise
        assert session[store._HYDRATED] is True

    def test_unknown_keys_are_ignored(self, session, monkeypatch):
        _replies(monkeypatch, {"ready": True,
                               "data": _blob(portfolio=[], evil="payload")})
        store.sync_browser()
        assert "_store_evil" not in session

    def test_a_non_dict_reply_is_ignored(self, session, monkeypatch):
        _replies(monkeypatch, "surprise")
        store.sync_browser()
        assert not session.get(store._HYDRATED)


class TestForget:
    def test_forget_clears_session_and_browser(self, session, monkeypatch):
        session["_store_portfolio"] = [{"symbol": "NVDA"}]
        session["_store_watchlist"] = [{"symbol": "AMD"}]
        calls = _replies(monkeypatch, None)
        store.forget_browser()
        assert calls[0]["clear"] is True
        assert "_store_portfolio" not in session
        assert "_store_watchlist" not in session


class TestLocalMode:
    def test_a_laptop_does_not_use_the_browser_bridge(self, monkeypatch):
        # Running locally, state belongs in data/ - mounting a component there
        # would be pointless work on every rerun.
        monkeypatch.setattr(store, "is_shared_host", lambda: False)
        calls = _replies(monkeypatch, {"ready": True, "data": None})
        store.sync_browser()
        assert calls == []
