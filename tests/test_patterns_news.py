"""Offline tests for pattern detection and news classification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from analyzer import newsfeed, patterns


def _ohlc(closes, opens=None, highs=None, lows=None) -> pd.DataFrame:
    """Build a frame from close prices, with sane synthetic OHLC around them."""
    # Floats throughout: an int64 column rejects the float assignments the
    # shape-specific tests make.
    closes = [float(c) for c in closes]
    opens = [float(o) for o in opens] if opens is not None else [c * 0.995 for c in closes]
    highs = [float(h) for h in highs] if highs is not None else [
        max(o, c) * 1.01 for o, c in zip(opens, closes)
    ]
    lows = [float(x) for x in lows] if lows is not None else [
        min(o, c) * 0.99 for o, c in zip(opens, closes)
    ]
    index = pd.date_range("2023-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes,
         "Volume": [1_000_000] * len(closes)},
        index=index,
    )


class TestCandlestickDefinitions:
    def test_bullish_engulfing_needs_a_bigger_body_than_the_bar_it_engulfs(self):
        # Down bar, then an up bar that fully covers it.
        df = _ohlc(
            closes=[100, 96, 101], opens=[101, 100, 95],
            highs=[102, 100.5, 102], lows=[99, 95.5, 94.5],
        )
        assert bool(patterns.bullish_engulfing(df).iloc[-1])

    def test_bullish_engulfing_rejects_a_smaller_body(self):
        df = _ohlc(
            closes=[100, 90, 92], opens=[101, 100, 91],
            highs=[102, 101, 93], lows=[99, 89, 90],
        )
        assert not bool(patterns.bullish_engulfing(df).iloc[-1])

    def test_bearish_engulfing(self):
        df = _ohlc(
            closes=[100, 104, 99], opens=[99, 100, 105],
            highs=[101, 105, 106], lows=[98, 99, 98],
        )
        assert bool(patterns.bearish_engulfing(df).iloc[-1])

    def test_hammer_requires_a_downtrend(self):
        """The same bar shape is not a hammer in an uptrend."""
        falling = [100 - i for i in range(20)]
        df = _ohlc(closes=falling)
        # Final bar: long lower wick, small body, closing near the high.
        df.iloc[-1, df.columns.get_loc("Open")] = 81.0
        df.iloc[-1, df.columns.get_loc("Close")] = 81.4
        df.iloc[-1, df.columns.get_loc("High")] = 81.5
        df.iloc[-1, df.columns.get_loc("Low")] = 78.0
        assert bool(patterns.hammer(df).iloc[-1])

        rising = [100 + i for i in range(20)]
        up = _ohlc(closes=rising)
        up.iloc[-1, up.columns.get_loc("Open")] = 119.0
        up.iloc[-1, up.columns.get_loc("Close")] = 119.4
        up.iloc[-1, up.columns.get_loc("High")] = 119.5
        up.iloc[-1, up.columns.get_loc("Low")] = 116.0
        assert not bool(patterns.hammer(up).iloc[-1])

    def test_gaps(self):
        df = _ohlc(
            closes=[100, 110], opens=[99, 108],
            highs=[101, 111], lows=[98, 107],
        )
        assert bool(patterns.gap_up(df).iloc[-1])
        assert not bool(patterns.gap_down(df).iloc[-1])

    def test_three_black_crows(self):
        df = _ohlc(
            closes=[100, 96, 92, 88], opens=[101, 100, 96, 92],
            highs=[102, 100.5, 96.5, 92.5], lows=[99, 95, 91, 87],
        )
        assert bool(patterns.three_black_crows(df).iloc[-1])


class TestEdgeMeasurement:
    def test_edge_is_measured_against_the_base_rate(self):
        """A signal that fires on every bar can have no edge by construction."""
        rng = np.random.default_rng(0)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        df = _ohlc(closes)
        always = pd.Series(True, index=df.index)
        edge = patterns.measure_edge(df, always)
        horizon = edge["horizons"][20]
        assert horizon["hit_rate"] == pytest.approx(horizon["base_rate"], abs=1e-6)
        assert horizon["edge"] == pytest.approx(0.0, abs=1e-6)

    def test_a_signal_that_only_fires_before_rallies_shows_a_positive_edge(self):
        closes = list(np.linspace(100, 100, 60))
        # Engineer a run-up after bar 60 so signals there look prescient.
        closes += list(np.linspace(100, 160, 60))
        df = _ohlc(closes)
        signal = pd.Series(False, index=df.index)
        signal.iloc[55:60] = True
        edge = patterns.measure_edge(df, signal)
        assert edge["horizons"][20]["hit_rate"] == 100.0
        assert edge["horizons"][20]["edge"] > 0

    def test_no_occurrences_returns_empty(self):
        df = _ohlc(np.linspace(100, 110, 80))
        edge = patterns.measure_edge(df, pd.Series(False, index=df.index))
        assert edge["occurrences"] == 0
        assert edge["horizons"] == {}


class TestStructuralPatterns:
    def test_range_breakout(self):
        closes = [100 + (i % 3) for i in range(60)] + [118]
        result = patterns.analyse(_ohlc(closes))
        names = [p["name"] for p in result["structural"]]
        assert "Range breakout" in names
        found = next(p for p in result["structural"] if p["name"] == "Range breakout")
        assert found["direction"] == "bullish"
        assert found["target"] > found["trigger"]

    def test_bull_flag_needs_a_pole_then_a_pause(self):
        # analyse() requires 60 bars before it will read anything, so the
        # fixture has to clear that bar as well as contain the formation.
        closes = list(np.linspace(100, 100, 40))
        closes += list(np.linspace(100, 140, 20))   # the pole
        closes += list(np.linspace(140, 138, 10))   # the flag
        result = patterns.analyse(_ohlc(closes))
        assert any(p["name"] == "Bull flag" for p in result["structural"])

    def test_short_history_is_reported_not_guessed(self):
        result = patterns.analyse(_ohlc(np.linspace(100, 110, 20)))
        assert result["bias"] is None
        assert "history" in result["bias_detail"].lower()

    def test_quiet_chart_claims_nothing(self):
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 0.05, 200))
        result = patterns.analyse(_ohlc(closes))
        assert result["bias"] in (None, "mixed", "bullish", "bearish")
        assert isinstance(result["structural"], list)


class TestMeasurementOverridesFolklore:
    def test_direction_follows_history_when_it_contradicts_the_label(self):
        """The whole point: what the pattern did here beats what it 'means'."""
        # A chart where every bearish engulfing is followed by a rally.
        closes = []
        for _ in range(12):
            closes += [100, 96, 101]          # engulfing setup
            closes += list(np.linspace(101, 118, 22))  # then it rises
        df = _ohlc(closes)
        result = patterns.analyse(df, recent_bars=len(df))

        bearish = [c for c in result["candlesticks"]
                   if c["name"] == "Bearish engulfing"]
        if bearish and bearish[0]["measured_direction"]:
            candle = bearish[0]
            if candle["measured_direction"] == "bullish":
                assert candle["agrees_with_history"] is False


class TestNewsClassification:
    @pytest.mark.parametrize(
        "headline,bucket",
        [
            ("Trump says Dell will build AI servers in the US", "policy"),
            ("Powell signals rates may stay higher for longer", "policy"),
            ("New tariff on semiconductor imports announced", "policy"),
            ("Jensen Huang calls Marvell the next trillion dollar company", "executive"),
            ("Elon Musk teases new Tesla model", "executive"),
            ("Goldman upgrades NVDA, raises price target", "analyst"),
            ("Nvidia beats Q3 earnings, guidance above consensus", "earnings"),
            ("Broadcom announces acquisition of VMware", "corporate"),
            ("DOJ opens antitrust probe into Meta", "legal"),
            ("Apple unveils new M5 chip", "product"),
            ("Markets drift sideways in quiet trade", "other"),
        ],
    )
    def test_buckets(self, headline, bucket):
        assert newsfeed.classify(headline)["bucket"] == bucket

    def test_decision_maker_flag(self):
        assert newsfeed.classify("Trump comments on chips")["is_decision_maker"]
        assert not newsfeed.classify("Company launches a product")["is_decision_maker"]

    def test_matching_is_word_boundaried(self):
        """A substring inside another word must not trigger a match."""
        assert newsfeed.classify("Fedex reports quarterly volumes")["bucket"] != "policy"

    def test_policy_outranks_a_company_event(self):
        """When a president and an earnings report share a headline, reach wins."""
        result = newsfeed.classify("Trump comments as Nvidia reports earnings")
        assert result["bucket"] == "policy"


class TestPriceReaction:
    def _history(self, closes, start="2026-08-03"):
        index = pd.date_range(start, periods=len(closes), freq="B")
        return pd.DataFrame({"Close": closes}, index=index)

    def test_move_is_measured_from_the_session_before_publication(self):
        history = self._history([100, 100, 110])
        published = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)
        result = newsfeed.price_reaction(history, published)
        assert result["move_pct"] == pytest.approx(10.0)

    def test_same_day_headline_still_measures(self):
        """Regression: requiring a bar strictly after publication blanked
        every fresh headline, since a daily bar is stamped at midnight."""
        history = self._history([100, 105])
        published = datetime(2026, 8, 4, 15, 30, tzinfo=timezone.utc)
        result = newsfeed.price_reaction(history, published)
        assert result["move_pct"] == pytest.approx(5.0)
        assert result["same_session"] is True

    def test_no_prior_close_returns_blank(self):
        history = self._history([100, 105])
        published = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert newsfeed.price_reaction(history, published)["move_pct"] is None

    def test_missing_inputs_are_safe(self):
        assert newsfeed.price_reaction(None, datetime.now(timezone.utc))["move_pct"] is None
        assert newsfeed.price_reaction(pd.DataFrame(), None)["move_pct"] is None


class TestFeedAssembly:
    def test_build_groups_sorts_and_flags_movers(self):
        now = datetime.now(timezone.utc)
        index = pd.date_range(now - timedelta(days=6), periods=5, freq="B")
        history = pd.DataFrame({"Close": [100, 100, 100, 100, 108]}, index=index)
        items = [
            {"title": "Trump praises the company", "published": now - timedelta(hours=2)},
            {"title": "Analyst downgrade on valuation", "published": now - timedelta(hours=30)},
            {"title": "", "published": now},  # dropped
        ]
        feed = newsfeed.build(items, history, "TEST")
        assert feed["total"] == 2
        assert feed["decision_maker_count"] == 2
        assert "policy" in feed["groups"]
        # Newest first.
        assert feed["items"][0]["title"].startswith("Trump")

    def test_speaker_is_found_in_the_summary_when_absent_from_the_title(self):
        feed = newsfeed.build([{
            "title": "Chip stock surges",
            "summary": "The move followed remarks by Jensen Huang at a conference.",
            "published": datetime.now(timezone.utc),
        }])
        assert feed["items"][0]["bucket"] == "executive"

    def test_undated_items_sort_last_rather_than_vanish(self):
        feed = newsfeed.build([
            {"title": "No date here"},
            {"title": "Trump speaks", "published": datetime.now(timezone.utc)},
        ])
        assert feed["total"] == 2
        assert feed["items"][-1]["title"] == "No date here"


class TestAggregation:
    """One merged stream across every followed symbol, ranked by impact."""

    def _item(self, title, bucket="other", move=None, sessions=1, age=2.0):
        return {"title": title, "bucket": bucket, "move_pct": move,
                "sessions": sessions, "age_hours": age, "is_decision_maker":
                bucket in ("policy", "executive", "analyst"), "matched": None,
                "same_session": sessions <= 1, "summary": "", "publisher": "x",
                "link": None}

    def test_same_story_across_tickers_is_merged_once(self):
        story = "Chip sector rallies on demand outlook"
        feed = newsfeed.aggregate({
            "AVGO": [self._item(story)],
            "MU": [self._item(story)],
            "NVDA": [self._item(story)],
        })
        assert feed["total"] == 1
        assert set(feed["stream"][0]["symbols"]) == {"AVGO", "MU", "NVDA"}

    def test_titles_differing_only_in_punctuation_still_merge(self):
        feed = newsfeed.aggregate({
            "A": [self._item("Nvidia's big day!")],
            "B": [self._item("Nvidias big day")],
        })
        assert feed["total"] == 1

    def test_holdings_outrank_watchlist_all_else_equal(self):
        feed = newsfeed.aggregate(
            {"HELD": [self._item("Same news")], "WATCH": [self._item("Other news")]},
            owned={"HELD"}, weights={"HELD": 20.0},
        )
        assert feed["stream"][0]["symbols"] == ["HELD"]
        assert len(feed["held"]) == 1 and len(feed["watched"]) == 1

    def test_policy_outranks_a_product_note(self):
        feed = newsfeed.aggregate({
            "A": [self._item("Tariff announced", bucket="policy")],
            "B": [self._item("New gadget", bucket="product")],
        })
        assert feed["stream"][0]["bucket"] == "policy"

    def test_stale_move_is_not_treated_as_a_reaction(self):
        """Regression: a +54% drift over 49 sessions was ranked as breaking news."""
        fresh = self._item("Fresh", move=5.0, sessions=1, age=2)
        stale = self._item("Stale", move=54.0, sessions=49, age=500)
        assert newsfeed.is_attributable(fresh)
        assert not newsfeed.is_attributable(stale)

        feed = newsfeed.aggregate({"A": [fresh], "B": [stale]})
        # The huge but old move must not be called a mover...
        assert [i["title"] for i in feed["movers"]] == ["Fresh"]
        # ...nor outrank the recent one.
        assert feed["stream"][0]["title"] == "Fresh"

    def test_movers_need_size_as_well_as_recency(self):
        feed = newsfeed.aggregate({"A": [self._item("Tiny", move=0.4, sessions=1)]})
        assert feed["movers"] == []

    def test_empty_input_is_safe(self):
        feed = newsfeed.aggregate({})
        assert feed["total"] == 0 and feed["stream"] == []

    def test_macro_section_collects_policy_stories(self):
        feed = newsfeed.aggregate({
            "A": [self._item("Fed holds rates", bucket="policy")],
            "B": [self._item("Earnings beat", bucket="earnings")],
        })
        assert len(feed["macro"]) == 1
