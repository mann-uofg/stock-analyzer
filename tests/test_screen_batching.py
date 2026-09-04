"""Tests for incremental screening and multi-symbol watchlist entry.

The bug these cover: screen results were memoised on the whole tuple of
symbols, so adding one name changed the cache key and re-analysed every other
name from scratch. Every added symbol felt like a full reload, the work grew
with the square of the list, and a long enough watchlist could not finish
inside a free container's memory. Once the watchlist persisted, that failure
repeated on every load and no refresh could clear it.
"""

from __future__ import annotations

import pytest

from analyzer import portfolio as pf, store
from views import common


@pytest.fixture(autouse=True)
def clean_cache():
    common.screen.clear()
    yield
    common.screen.clear()


@pytest.fixture
def counted(monkeypatch):
    """Record which symbols actually get analysed."""
    calls: list[str] = []

    def fake_one(symbol: str, period: str) -> dict:
        calls.append(symbol)
        return {"symbol": symbol, "price": 100.0}

    monkeypatch.setattr(common, "_screen_one", fake_one)
    return calls


class TestIncrementalScreening:
    def test_adding_one_symbol_analyses_only_that_symbol(self, counted):
        # The reported bug: this used to re-analyse all four.
        common.screen(("NVDA", "AMD", "TSLA"))
        counted.clear()
        common.screen(("NVDA", "AMD", "TSLA", "SHOP.TO"))
        assert counted == ["SHOP.TO"]

    def test_a_repeat_screen_costs_nothing(self, counted):
        common.screen(("NVDA", "AMD"))
        counted.clear()
        common.screen(("NVDA", "AMD"))
        assert counted == []

    def test_reordering_does_not_invalidate(self, counted):
        common.screen(("NVDA", "AMD"))
        counted.clear()
        common.screen(("AMD", "NVDA"))
        assert counted == []

    def test_results_keep_the_callers_order(self, counted):
        rows = common.screen(("TSLA", "AMD", "NVDA"))
        assert [r["symbol"] for r in rows] == ["TSLA", "AMD", "NVDA"]

    def test_clear_forces_a_full_recompute(self, counted):
        common.screen(("NVDA",))
        common.screen.clear()
        counted.clear()
        common.screen(("NVDA",))
        assert counted == ["NVDA"]

    def test_failures_are_not_cached(self, monkeypatch):
        # A symbol that failed on a network blip must be retried, not held as
        # broken for the rest of the hour.
        calls: list[str] = []

        def flaky(symbol: str, period: str) -> dict:
            calls.append(symbol)
            if len(calls) == 1:
                return {"symbol": symbol, "error": "timeout"}
            return {"symbol": symbol, "price": 42.0}

        monkeypatch.setattr(common, "_screen_one", flaky)
        common.screen(("NVDA",))
        rows = common.screen(("NVDA",))
        assert calls == ["NVDA", "NVDA"]
        assert rows[0].get("price") == 42.0


class TestBatchLimit:
    def test_limit_caps_new_work_per_pass(self, counted):
        common.screen(tuple(f"S{i}" for i in range(10)), limit=4)
        assert len(counted) == 4

    def test_batches_accumulate_across_passes(self, counted):
        symbols = tuple(f"S{i}" for i in range(10))
        common.screen(symbols, limit=4)
        common.screen(symbols, limit=4)
        common.screen(symbols, limit=4)
        assert len(counted) == 10
        assert len(common.screen(symbols)) == 10

    def test_cached_rows_are_returned_even_when_capped(self, counted):
        symbols = ("A", "B", "C", "D")
        common.screen(symbols, limit=2)
        rows = common.screen(symbols, limit=0)
        # The two already analysed still render; nothing new is computed.
        assert len(rows) == 2

    def test_pending_reports_what_is_left(self, counted):
        symbols = ("A", "B", "C")
        common.screen(symbols, limit=1)
        assert len(common.screen_pending(symbols)) == 2

    def test_no_limit_does_everything(self, counted):
        common.screen(tuple(f"S{i}" for i in range(7)))
        assert len(counted) == 7


class TestParseSymbolList:
    @pytest.mark.parametrize("text,expected", [
        ("NVDA, AMD, TSLA", ["NVDA", "AMD", "TSLA"]),
        ("NVDA AMD", ["NVDA", "AMD"]),
        ("NVDA\nAMD\nTSLA", ["NVDA", "AMD", "TSLA"]),
        ("nvda, amd", ["NVDA", "AMD"]),
        ("NVDA;AMD", ["NVDA", "AMD"]),
        ("  NVDA ,  AMD  ", ["NVDA", "AMD"]),
        ("NVDA,,AMD", ["NVDA", "AMD"]),
    ])
    def test_separators(self, text, expected):
        assert pf.parse_symbol_list(text) == expected

    def test_exchange_suffixes_survive(self):
        assert pf.parse_symbol_list("SHOP.TO, BRK-B") == ["SHOP.TO", "BRK-B"]

    def test_a_trailing_dot_is_trimmed(self):
        assert pf.parse_symbol_list("NVDA.") == ["NVDA"]

    def test_duplicates_within_a_paste_are_dropped(self):
        assert pf.parse_symbol_list("NVDA, AMD, NVDA") == ["NVDA", "AMD"]

    @pytest.mark.parametrize("text", ["", "   ", ",,,", None])
    def test_empty_input_yields_nothing(self, text):
        assert pf.parse_symbol_list(text) == []


class TestAddMany:
    @pytest.fixture
    def session(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(store, "is_shared_host", lambda: True)
        monkeypatch.setattr(store, "_session_state", lambda: state)
        return state

    def test_all_symbols_land_in_one_write(self, session):
        added, skipped = store.add_many_to_watchlist(["NVDA", "AMD", "TSLA"])
        assert added == ["NVDA", "AMD", "TSLA"]
        assert len(store.load_watchlist()) == 3

    def test_existing_symbols_are_reported_not_duplicated(self, session):
        store.add_many_to_watchlist(["NVDA"])
        added, skipped = store.add_many_to_watchlist(["NVDA", "AMD"])
        assert added == ["AMD"] and skipped == ["NVDA"]
        assert len(store.load_watchlist()) == 2

    def test_duplicates_inside_one_call_are_collapsed(self, session):
        added, _ = store.add_many_to_watchlist(["NVDA", "NVDA"])
        assert added == ["NVDA"]

    def test_nothing_new_means_no_write(self, session):
        store.add_many_to_watchlist(["NVDA"])
        session[store._DIRTY] = False
        store.add_many_to_watchlist(["NVDA"])
        assert session[store._DIRTY] is False

    def test_blanks_are_ignored(self, session):
        added, _ = store.add_many_to_watchlist(["NVDA", "", "  "])
        assert added == ["NVDA"]

    def test_the_note_is_applied_to_each(self, session):
        store.add_many_to_watchlist(["NVDA", "AMD"], note="AI basket")
        assert all(e["note"] == "AI basket" for e in store.load_watchlist())
