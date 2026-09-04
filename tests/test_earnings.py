"""Tests for the earnings calendar and its expectation maths.

The load-bearing test here is the implied-move basis. A straddle prices every
day between now and expiry, so with a report ten weeks out it measures ten
weeks of ordinary movement with one event buried in it - NVIDIA's read 14%
against a company that typically moves 3% on results. Presenting that as the
earnings expectation would have been wrong by a factor of four, so the code
must refuse to call it one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from analyzer import earnings


def _frame(rows):
    """An earnings frame shaped like yfinance's, newest first."""
    index = pd.DatetimeIndex([r[0] for r in rows], tz="America/New_York")
    return pd.DataFrame(
        {
            "EPS Estimate": [r[1] for r in rows],
            "Reported EPS": [r[2] for r in rows],
            "Surprise(%)": [r[3] for r in rows],
        },
        index=index,
    )


def _prices(start: str, days: int = 400, step: float = 0.5):
    idx = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"Close": np.arange(100.0, 100.0 + days * step, step)[:days]},
                        index=idx)


class TestSessionClassification:
    @pytest.mark.parametrize("hour,minute,expected", [
        (7, 0, earnings.BEFORE), (9, 0, earnings.BEFORE),
        (16, 0, earnings.AFTER), (16, 30, earnings.AFTER),
        (20, 0, earnings.AFTER),
    ])
    def test_clear_times(self, hour, minute, expected):
        stamp = pd.Timestamp(f"2026-11-17 {hour:02d}:{minute:02d}",
                             tz="America/New_York")
        assert earnings.classify_session(stamp) == expected

    @pytest.mark.parametrize("hour", [10, 12, 15])
    def test_midsession_times_are_not_asserted(self, hour):
        # Yahoo stamps NVIDIA's next report at 15:00 despite it always
        # reporting after the close. A time inside market hours is a
        # placeholder, and announcing a mid-session report would be a
        # fabrication.
        stamp = pd.Timestamp(f"2026-11-17 {hour:02d}:00", tz="America/New_York")
        assert earnings.classify_session(stamp) == earnings.UNKNOWN

    def test_missing_timestamp_is_unknown(self):
        assert earnings.classify_session(None) == earnings.UNKNOWN
        assert earnings.classify_session("not a time") == earnings.UNKNOWN


class TestUpcoming:
    def test_picks_the_nearest_unreported_date(self):
        df = _frame([
            ("2026-11-17 16:00", 2.47, np.nan, np.nan),
            ("2026-08-26 16:00", 2.09, 2.22, 6.16),
        ])
        event = earnings.upcoming("NVDA", df, {}, today=dt.date(2026, 9, 4))
        assert event["date"] == "2026-11-17"
        assert event["days_away"] == 74
        assert event["session"] == earnings.AFTER

    def test_already_reported_dates_are_skipped(self):
        df = _frame([("2026-09-05 16:00", 1.0, 1.1, 10.0)])
        assert earnings.upcoming("X", df, {}, today=dt.date(2026, 9, 4)) is None

    def test_past_dates_are_skipped(self):
        df = _frame([("2026-08-01 16:00", 1.0, np.nan, np.nan)])
        assert earnings.upcoming("X", df, {}, today=dt.date(2026, 9, 4)) is None

    def test_consensus_range_becomes_a_spread(self):
        df = _frame([("2026-11-17 16:00", 2.47, np.nan, np.nan)])
        cal = {"Earnings High": 2.7, "Earnings Low": 2.34,
               "Earnings Average": 2.47}
        event = earnings.upcoming("NVDA", df, cal, today=dt.date(2026, 9, 4))
        assert event["eps_spread_pct"] == pytest.approx(14.57, abs=0.1)

    def test_calendar_alone_is_enough(self):
        cal = {"Earnings Date": [dt.date(2026, 10, 30)], "Earnings Average": 1.98}
        event = earnings.upcoming("AAPL", pd.DataFrame(), cal,
                                  today=dt.date(2026, 9, 4))
        assert event["date"] == "2026-10-30"
        assert event["eps_estimate"] == 1.98

    def test_a_one_day_disagreement_is_flagged(self):
        # Yahoo's two sources routinely name consecutive dates for one event.
        df = _frame([("2026-11-17 16:00", 2.47, np.nan, np.nan)])
        cal = {"Earnings Date": [dt.date(2026, 11, 18)]}
        event = earnings.upcoming("NVDA", df, cal, today=dt.date(2026, 9, 4))
        assert event["date_disputed"] is True

    def test_nothing_scheduled_returns_none(self):
        assert earnings.upcoming("SPY", pd.DataFrame(), {}) is None


class TestPastEvents:
    def test_surprise_and_move_are_paired(self):
        df = _frame([
            ("2026-08-26 16:00", 2.09, 2.22, 6.16),
            ("2026-05-20 16:00", 1.77, 1.87, 5.54),
        ])
        events = earnings.past_events(df, _prices("2026-05-01"))
        assert len(events) == 2
        assert events[0]["surprise_pct"] == 6.16
        assert events[0]["move_pct"] is not None
        assert events[0]["beat"] is True

    def test_unreported_rows_are_excluded(self):
        df = _frame([
            ("2026-11-17 16:00", 2.47, np.nan, np.nan),
            ("2026-08-26 16:00", 2.09, 2.22, 6.16),
        ])
        events = earnings.past_events(df, _prices("2026-05-01"))
        assert [e["date"] for e in events] == ["2026-08-26"]

    def test_surprise_is_derived_when_absent(self):
        df = pd.DataFrame(
            {"EPS Estimate": [2.0], "Reported EPS": [2.2]},
            index=pd.DatetimeIndex(["2026-08-26"], tz="America/New_York"),
        )
        events = earnings.past_events(df, None)
        assert events[0]["surprise_pct"] == pytest.approx(10.0)

    def test_missing_prices_leave_the_move_empty(self):
        df = _frame([("2026-08-26 16:00", 2.09, 2.22, 6.16)])
        assert earnings.past_events(df, None)[0]["move_pct"] is None

    def test_empty_input_is_safe(self):
        assert earnings.past_events(pd.DataFrame(), None) == []
        assert earnings.past_events(None, None) == []


class TestHistoryStats:
    def _events(self):
        return [
            {"surprise_pct": 6.0, "move_pct": 8.0},
            {"surprise_pct": 5.0, "move_pct": -2.0},
            {"surprise_pct": -1.0, "move_pct": 3.0},
            {"surprise_pct": 4.0, "move_pct": -25.0},
        ]

    def test_beat_rate(self):
        assert earnings.history_stats(self._events())["beat_rate_pct"] == 75.0

    def test_typical_move_is_a_median_not_a_mean(self):
        # One 25% quarter must not set the expectation for a stock that
        # usually moves 3%.
        stats = earnings.history_stats(self._events())
        assert stats["typical_move_pct"] == pytest.approx(5.5)

    def test_largest_move_keeps_its_sign(self):
        assert earnings.history_stats(self._events())["largest_move_pct"] == -25.0

    def test_up_rate(self):
        assert earnings.history_stats(self._events())["up_rate_pct"] == 50.0

    def test_empty_history_is_all_none(self):
        stats = earnings.history_stats([])
        assert stats["quarters"] == 0
        assert stats["beat_rate_pct"] is None


class TestImpliedMoveBasis:
    """The correctness guard: is this straddle actually about the report?"""

    @pytest.fixture
    def chain(self, monkeypatch):
        def expiries(symbol):
            return ["2026-09-11", "2026-11-20", "2026-12-18"]

        def chain_for(symbol, expiry):
            calls = pd.DataFrame({"strike": [220.0, 230.0, 240.0],
                                  "bid": [20.0, 16.0, 12.0],
                                  "ask": [21.0, 17.0, 13.0],
                                  "lastPrice": [20.5, 16.5, 12.5]})
            puts = calls.copy()
            return calls, puts

        monkeypatch.setattr(earnings.datafeed, "option_expirations", expiries)
        monkeypatch.setattr(earnings.datafeed, "option_chain", chain_for)
        monkeypatch.setattr(earnings.datafeed, "fast_quote",
                            lambda s: {"last_price": 230.0})

    def test_far_out_report_is_labelled_period_not_event(self, chain):
        # Report 77 days away, expiry 3 days after it: the premium is mostly
        # ten weeks of ordinary movement.
        out = earnings.implied_move("NVDA", "2026-11-17", today=dt.date(2026, 9, 4))
        assert out["basis"] == "period"

    def test_near_report_with_tight_expiry_is_an_event_reading(self, chain):
        out = earnings.implied_move("NVDA", "2026-11-17",
                                    today=dt.date(2026, 11, 10))
        assert out["basis"] == "event"
        assert out["gap_days"] == 3

    def test_expiry_long_after_the_report_is_never_an_event_reading(self, chain):
        # AAPL's case: the only expiry sits 22 days past the report.
        out = earnings.implied_move("AAPL", "2026-10-29",
                                    today=dt.date(2026, 10, 27))
        assert out["gap_days"] == 22
        assert out["basis"] == "period"

    def test_move_is_the_straddle_over_spot(self, chain):
        out = earnings.implied_move("NVDA", "2026-11-17", today=dt.date(2026, 9, 4))
        # 16.5 call + 16.5 put over a 230 spot.
        assert out["move_pct"] == pytest.approx(33.0 / 230.0 * 100, abs=0.01)

    def test_an_expiry_before_the_report_is_not_used(self, chain):
        out = earnings.implied_move("NVDA", "2026-11-17", today=dt.date(2026, 9, 4))
        assert out["expiry"] == "2026-11-20"

    def test_no_expiry_covers_the_report(self, chain):
        assert earnings.implied_move("NVDA", "2027-06-01",
                                     today=dt.date(2026, 9, 4)) is None

    def test_a_bad_date_returns_nothing(self, chain):
        assert earnings.implied_move("NVDA", "not-a-date") is None


class TestBriefRatio:
    def test_ratio_is_withheld_when_the_straddle_is_not_about_the_event(
            self, monkeypatch):
        # Comparing a ten-week range against a one-day earnings move would
        # manufacture alarm out of pure time value.
        monkeypatch.setattr(earnings.datafeed, "earnings_history",
                            lambda s: _frame([("2026-11-17 16:00", 2.47,
                                               np.nan, np.nan)]))
        monkeypatch.setattr(earnings.datafeed, "earnings_calendar", lambda s: {})
        monkeypatch.setattr(earnings.datafeed, "price_history",
                            lambda s, period="3y": _prices("2025-06-01"))
        monkeypatch.setattr(
            earnings, "implied_move",
            lambda *a, **k: {"move_pct": 14.0, "basis": "period",
                             "expiry": "2026-11-20", "days_to_expiry": 77,
                             "gap_days": 3, "straddle": 32.0, "strike": 230.0},
        )
        out = earnings.brief("NVDA")
        assert out["implied_vs_typical"] is None
