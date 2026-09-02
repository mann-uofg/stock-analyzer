"""Tests for the plain-language reading of each metric.

Two things matter here. Band edges must land on the side the convention says
they do - an RSI of exactly 70 is not yet overbought, and getting that wrong
would mislabel a reading at precisely the moment it starts to matter. And the
gauge geometry must be sound, because a marker in the wrong place is a picture
that contradicts its own number.
"""

from __future__ import annotations

import math

import pytest

from analyzer import interpret as I


class TestBandEdges:
    @pytest.mark.parametrize("value,verdict", [
        (12.0, "Oversold"), (29.9, "Oversold"), (30.0, "Oversold"),
        (30.1, "Soft"), (50.0, "Neutral"), (69.9, "Strong"),
        (70.0, "Strong"), (70.1, "Overbought"), (99.0, "Overbought"),
    ])
    def test_rsi_bands(self, value, verdict):
        assert I.read("rsi_14", value).verdict == verdict

    @pytest.mark.parametrize("value,verdict", [
        (0.4, "Very defensive"), (0.9, "Moves with the market"),
        (1.3, "Aggressive"), (2.1, "Very aggressive"),
    ])
    def test_beta_bands(self, value, verdict):
        assert I.read("beta", value).verdict == verdict

    def test_adx_below_20_is_no_trend(self):
        assert I.read("adx_14", 15).verdict == "No real trend"

    def test_drawdown_bands_run_the_right_way(self):
        # A more negative drawdown is worse, so the bands ascend through
        # negative space - easy to get backwards.
        assert I.read("max_drawdown_1y_pct", -45).verdict == "Brutal"
        assert I.read("max_drawdown_1y_pct", -5).verdict == "Mild"

    def test_negative_fcf_yield_is_burning_cash(self):
        r = I.read("fcf_yield_pct", -3.0)
        assert r.verdict == "Burning cash" and r.tone == I.BAD


class TestBetaSentence:
    """The phrasing that does the real explanatory work."""

    def test_value_is_quoted_back_in_plain_english(self):
        assert "1.74% for every 1%" in I.read("beta", 1.74).plain

    def test_negative_beta_is_described_by_magnitude(self):
        # "-0.30% for every 1% move" would read as nonsense; the sentence
        # takes the magnitude.
        assert "-" not in I.read("beta", -0.3).plain.split("for every")[0]


class TestMissingAndBrokenValues:
    def test_none_is_reported_not_guessed(self):
        r = I.read("sharpe", None)
        assert r.value is None and r.display == "—" and not r.ok

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_is_not_meaningful(self, bad):
        r = I.read("peg", bad)
        assert r.verdict == "Not meaningful" and not r.ok

    def test_unknown_metric_returns_none_rather_than_inventing(self):
        assert I.read("not_a_real_metric", 1.0) is None

    def test_non_numeric_returns_none(self):
        assert I.read("rsi_14", "banana") is None

    def test_read_many_drops_unknowns(self):
        out = I.read_many([("rsi_14", 50), ("nonsense", 1)])
        assert len(out) == 1


class TestGaugeGeometry:
    @pytest.mark.parametrize("key", list(I.SPECS))
    def test_zones_tile_the_whole_track(self, key):
        z = I.zones(key)
        assert z, f"{key} has no zones"
        assert z[0][0] == pytest.approx(0.0)
        assert z[-1][1] == pytest.approx(1.0)
        for (_, end), (start, _) in zip(
            [(a, b) for a, b, _ in z], [(a, b) for a, b, _ in z][1:]
        ):
            assert end == pytest.approx(start), f"{key} has a gap or overlap"

    @pytest.mark.parametrize("key", list(I.SPECS))
    def test_every_spec_has_a_finite_scale(self, key):
        spec = I.SPECS[key]
        assert math.isfinite(spec.lo) and math.isfinite(spec.hi)
        assert spec.hi > spec.lo

    @pytest.mark.parametrize("key", list(I.SPECS))
    def test_bands_are_ordered(self, key):
        edges = [b.upto for b in I.SPECS[key].bands]
        assert edges == sorted(edges), f"{key} bands are out of order"
        assert edges[-1] == float("inf"), f"{key} has no catch-all band"

    def test_position_is_clamped_to_the_track(self):
        # A P/E of 900 must not push the marker off the end of the bar.
        assert I.read("trailing_pe", 900).position == pytest.approx(1.0)
        assert I.read("rsi_14", 0).position == pytest.approx(0.0)

    def test_marker_sits_where_the_value_does(self):
        assert I.read("rsi_14", 50).position == pytest.approx(0.5)


class TestContent:
    @pytest.mark.parametrize("key", list(I.SPECS))
    def test_every_band_explains_itself(self, key):
        for band in I.SPECS[key].bands:
            assert band.plain.strip(), f"{key}/{band.verdict} has no explanation"
            assert band.verdict.strip()
            assert band.tone in (I.GOOD, I.WARN, I.BAD, I.NEUTRAL)

    @pytest.mark.parametrize("key", list(I.SPECS))
    def test_every_metric_says_what_it_is(self, key):
        assert I.what_is(key).strip(), f"{key} has no plain-language gloss"

    def test_valuation_metrics_warn_about_sector(self):
        # The single most common way to misread a P/E.
        for key in ("trailing_pe", "forward_pe", "ev_ebitda"):
            assert "sector" in I.SPECS[key].note.lower()

    def test_adx_note_says_it_is_direction_blind(self):
        assert "never which way" in I.SPECS["adx_14"].note


class TestGrouping:
    def _mixed(self):
        return I.read_many([
            ("rsi_14", 50),          # neutral
            ("sharpe", 2.5),         # good
            ("beta", 2.0),           # bad
            ("adx_14", 15),          # neutral
            ("max_drawdown_1y_pct", -25),  # warn
        ])

    def test_concerns_lead_with_the_worst(self):
        flagged = I.concerns(self._mixed())
        assert flagged[0].tone == I.BAD

    def test_concerns_exclude_the_reassuring(self):
        assert all(r.tone != I.GOOD for r in I.concerns(self._mixed()))

    def test_encouraging_picks_only_good(self):
        assert all(r.tone == I.GOOD for r in I.encouraging(self._mixed()))

    def test_summary_counts_what_it_claims(self):
        text = I.summarise(self._mixed())
        assert "5" in text

    def test_summary_handles_no_data(self):
        assert "not enough data" in I.summarise(
            [I.read("sharpe", None)]).lower()

    def test_all_clear_summary_says_so(self):
        clean = I.read_many([("sharpe", 2.5), ("rsi_14", 60)])
        assert "nothing" in I.summarise(clean).lower()

    def test_mostly_neutral_says_so_rather_than_calling_it_mixed(self):
        # The live NVDA reading: five unremarkable measures and one flag.
        # Reporting that as "1 encouraging, 1 cautionary" buried the finding.
        live = I.read_many([
            ("atr_percent", 3.37), ("rsi_14", 57.1), ("stoch_k", 76.8),
            ("cci_20", 73), ("adx_14", 15.9), ("bb_percent_b", 0.77),
            ("volume_ratio", 0.85),
        ])
        text = I.summarise(live, "the chart")
        assert "unremarkable" in text and "5 of 7" in text
