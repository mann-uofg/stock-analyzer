"""Offline unit tests.

Nothing here touches the network: the data layer is exercised through
synthetic frames and a stub chain loader, so the suite is deterministic and
runs in under a second.

    .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analyzer import indicators, llm, options, risk, scoring
from analyzer.config import MIN_RISK_REWARD


# --- Fixtures -------------------------------------------------------------


def _ohlcv(n: int = 400, seed: int = 7, trend: float = 0.05) -> pd.DataFrame:
    """A synthetic but well-formed OHLCV frame."""
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(trend, 1.0, n)) + 100
    close = np.maximum(close, 1.0)
    high = close + rng.random(n)
    low = close - rng.random(n)
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, n),
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


# --- Black-Scholes --------------------------------------------------------


class TestBlackScholes:
    S, K, T, r, sigma = 217.5, 220.0, 45 / 365, 0.037, 0.42

    def test_put_call_parity(self):
        """C - P must equal S - K*e^{-rT}."""
        call = options.black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "c")
        put = options.black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "p")
        expected = self.S - self.K * math.exp(-self.r * self.T)
        assert call - put == pytest.approx(expected, abs=1e-9)

    def test_delta_parity(self):
        """With no dividend, call delta - put delta == 1."""
        c = options.black_scholes_greeks(self.S, self.K, self.T, self.r, self.sigma, "c")
        p = options.black_scholes_greeks(self.S, self.K, self.T, self.r, self.sigma, "p")
        assert c["delta"] - p["delta"] == pytest.approx(1.0, abs=1e-9)

    def test_gamma_and_vega_are_kind_independent(self):
        c = options.black_scholes_greeks(self.S, self.K, self.T, self.r, self.sigma, "c")
        p = options.black_scholes_greeks(self.S, self.K, self.T, self.r, self.sigma, "p")
        assert c["gamma"] == pytest.approx(p["gamma"], abs=1e-12)
        assert c["vega"] == pytest.approx(p["vega"], abs=1e-12)

    @pytest.mark.parametrize(
        "S,K,T,sigma",
        [(0, 100, 0.5, 0.3), (100, 0, 0.5, 0.3), (100, 100, 0, 0.3), (100, 100, 0.5, 0)],
    )
    def test_degenerate_inputs_return_none(self, S, K, T, sigma):
        """Degenerate inputs must yield None, never inf or nan."""
        out = options.black_scholes_greeks(S, K, T, 0.03, sigma, "c")
        assert all(v is None for v in out.values())

    def test_matches_reference_implementation(self):
        """Cross-check against py_vollib when it is installed."""
        pv = pytest.importorskip("py_vollib.black_scholes.greeks.analytical")
        bs = pytest.importorskip("py_vollib.black_scholes")
        for kind in ("c", "p"):
            mine = options.black_scholes_greeks(self.S, self.K, self.T, self.r, self.sigma, kind)
            args = (kind, self.S, self.K, self.T, self.r, self.sigma)
            assert mine["price"] == pytest.approx(bs.black_scholes(*args), abs=1e-9)
            assert mine["delta"] == pytest.approx(pv.delta(*args), abs=1e-9)
            assert mine["gamma"] == pytest.approx(pv.gamma(*args), abs=1e-9)
            assert mine["theta"] == pytest.approx(pv.theta(*args), abs=1e-9)
            assert mine["vega"] == pytest.approx(pv.vega(*args), abs=1e-9)


class TestImpliedVolatility:
    def test_round_trip(self):
        """Pricing at sigma then solving must recover sigma."""
        S, K, T, r = 100.0, 105.0, 0.25, 0.04
        for kind in ("c", "p"):
            for sigma in (0.10, 0.25, 0.40, 0.85, 1.5):
                price = options.black_scholes_price(S, K, T, r, sigma, kind)
                solved = options.implied_volatility(price, S, K, T, r, kind)
                assert solved == pytest.approx(sigma, abs=1e-4), f"{kind} @ {sigma}"

    def test_price_below_intrinsic_is_unsolvable(self):
        """A price at/below intrinsic has no implied vol; return None."""
        S, K, T, r = 150.0, 100.0, 0.5, 0.04
        intrinsic = S - K * math.exp(-r * T)
        assert options.implied_volatility(intrinsic * 0.9, S, K, T, r, "c") is None

    def test_garbage_inputs(self):
        assert options.implied_volatility(0, 100, 100, 0.5, 0.04, "c") is None
        assert options.implied_volatility(5, 100, 100, 0, 0.04, "c") is None
        assert options.implied_volatility(float("nan"), 100, 100, 0.5, 0.04, "c") is None


class TestChainRepair:
    """The feed-quality logic that rejects Yahoo's placeholder volatilities."""

    def _chain(self, quoted_iv: float) -> pd.DataFrame:
        spot, T, r = 100.0, 0.25, 0.04
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        true_iv = 0.35
        return pd.DataFrame(
            {
                "strike": strikes,
                "lastPrice": [
                    options.black_scholes_price(spot, k, T, r, true_iv, "c") for k in strikes
                ],
                "bid": [0.0] * len(strikes),
                "ask": [0.0] * len(strikes),
                "impliedVolatility": [quoted_iv] * len(strikes),
                "openInterest": [0] * len(strikes),
                "volume": [100] * len(strikes),
            }
        )

    def test_placeholder_iv_is_rejected_and_solved(self):
        """0.03126 clears a naive range check but cannot reprice the option."""
        cleaned = options._clean_chain(self._chain(0.03126), 100.0, 0.25, 0.04, "c")
        assert set(cleaned["iv_source"]) == {"solved"}
        assert cleaned["iv_used"].between(0.34, 0.36).all()

    def test_solved_iv_is_preferred_over_the_quote(self):
        """Even a correct quote is re-derived, keeping IV and Greeks consistent."""
        cleaned = options._clean_chain(self._chain(0.35), 100.0, 0.25, 0.04, "c")
        assert set(cleaned["iv_source"]) == {"solved"}
        assert cleaned["iv_used"].between(0.349, 0.351).all()

    def test_quote_is_used_when_price_is_unusable(self):
        """With no solvable price, a plausible quote is the remaining fallback."""
        chain = self._chain(0.35)
        chain["lastPrice"] = 0.0  # nothing to invert
        cleaned = options._clean_chain(chain, 100.0, 0.25, 0.04, "c")
        assert set(cleaned["iv_source"]) == {"yahoo"}
        assert cleaned["iv_used"].eq(0.35).all()

    def test_deep_itm_placeholder_is_still_rejected(self):
        """The case that defeats price-repricing validation: tiny vega.

        At K=90 with spot 100 the option is almost all intrinsic, so a
        placeholder IV reprices within tolerance yet implies wrong Greeks.
        Solving from price must recover the true volatility anyway.
        """
        chain = self._chain(0.03126).iloc[[0]]  # K=90 only
        cleaned = options._clean_chain(chain, 100.0, 0.25, 0.04, "c")
        assert cleaned["iv_source"].iloc[0] == "solved"
        assert cleaned["iv_used"].iloc[0] == pytest.approx(0.35, abs=1e-3)

    def test_analyse_reports_missing_open_interest(self):
        calls, puts = self._chain(0.03126), self._chain(0.03126)
        result = options.analyse(
            "TEST", 100.0, ["2099-01-01"], lambda t, e: (calls, puts), 0.04,
        )
        assert result["available"]
        assert result["data_quality"]["open_interest_available"] is False
        # An unavailable ratio must be None, never a misleading zero.
        assert result["put_call_ratio"]["open_interest"] is None
        assert result["gamma_exposure"]["weighted_by"] == "volume"

    def test_no_expiries_is_graceful(self):
        out = options.analyse("TEST", 100.0, [], lambda t, e: (pd.DataFrame(), pd.DataFrame()), 0.04)
        assert out["available"] is False and "reason" in out


# --- Indicators -----------------------------------------------------------


class TestIndicators:
    def test_full_panel(self):
        out = indicators.compute(_ohlcv())
        assert out["moving_averages"]["sma_20"] is not None
        assert 0 <= out["momentum"]["rsi_14"] <= 100
        assert out["volatility"]["atr_14"] > 0
        assert out["levels"]["support"] or out["levels"]["resistance"]

    def test_short_history_does_not_raise(self):
        """Two bars cannot support a 200-day SMA; we must degrade, not crash."""
        out = indicators.compute(_ohlcv(n=3))
        assert out["moving_averages"]["sma_200"] is None
        assert out["price"] > 0

    def test_empty_frame(self):
        assert "error" in indicators.compute(pd.DataFrame())

    def test_golden_cross_detected(self):
        """A series that rises after falling must produce a golden cross."""
        n = 500
        values = np.concatenate([np.linspace(200, 100, n // 2), np.linspace(100, 260, n // 2)])
        df = pd.DataFrame(
            {"Open": values, "High": values + 1, "Low": values - 1, "Close": values,
             "Adj Close": values, "Volume": np.full(n, 1e6)},
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        cross = indicators.compute(df)["moving_averages"]["golden_death_cross"]
        assert cross["state"] == "golden"
        assert cross["event"] == "golden_cross"


# --- Risk -----------------------------------------------------------------


class TestRisk:
    def test_beta_of_identical_series_is_one(self):
        df = _ohlcv()
        out = risk.compute(df, {"SPY": df}, 0.04)
        one_year = out["benchmarks"]["SPY"]["1y"]
        assert one_year["beta"] == pytest.approx(1.0, abs=1e-9)
        assert one_year["r_squared"] == pytest.approx(1.0, abs=1e-9)
        # Against itself, CAPM leaves no excess return.
        assert one_year["alpha_annual_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_beta_scales_with_leverage(self):
        """A 2x-levered version of the benchmark must show beta ~2."""
        base = _ohlcv()
        rets = base["Close"].pct_change().fillna(0)
        levered = 100 * (1 + 2 * rets).cumprod()
        lev_df = pd.DataFrame(
            {"Open": levered, "High": levered, "Low": levered, "Close": levered,
             "Adj Close": levered, "Volume": base["Volume"].to_numpy()},
            index=base.index,
        )
        out = risk.compute(lev_df, {"SPY": base}, 0.04)
        assert out["benchmarks"]["SPY"]["1y"]["beta"] == pytest.approx(2.0, abs=0.02)

    def test_missing_benchmark_is_tolerated(self):
        out = risk.compute(_ohlcv(), {}, 0.04)
        assert out["benchmarks"] == {}
        assert out["volatility"]["hv_30d_annual_pct"] > 0

    def test_perfectly_correlated_series_is_not_flagged_weak(self):
        df = _ohlcv()
        out = risk.compute(df, {"SPY": df}, 0.04)
        assert out["benchmarks"]["SPY"]["1y"]["low_explanatory_power"] is False

    def test_uncorrelated_series_is_flagged_weak(self):
        """A beta off near-zero correlation must be marked statistically weak."""
        out = risk.compute(_ohlcv(seed=1), {"SPY": _ohlcv(seed=999)}, 0.04)
        one_year = out["benchmarks"]["SPY"]["1y"]
        assert one_year["r_squared"] < 0.10
        assert one_year["low_explanatory_power"] is True


# --- Scoring and trade construction --------------------------------------


class TestScoring:
    def _panels(self):
        df = _ohlcv()
        tech = indicators.compute(df)
        risk_panel = risk.compute(df, {"SPY": _ohlcv(seed=11)}, 0.04)
        fund = {"valuation": {}, "earnings": {"history": []}, "consensus": {}}
        return tech, risk_panel, {"available": False, "reason": "test"}, fund, float(df["Close"].iloc[-1])

    def test_output_ranges(self):
        out = scoring.compute(*self._panels())
        assert 0 <= out["score_0_100"] <= 100
        assert 0 <= out["conviction_pct"] <= 100
        assert out["verdict"] in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
        for bucket in out["buckets"].values():
            assert -1.0 <= bucket["score"] <= 1.0
            assert bucket["reasons"], "every bucket must explain itself"

    @pytest.mark.parametrize("direction", ["long", "short"])
    def test_trade_setup_invariants(self, direction):
        tech = indicators.compute(_ohlcv())
        price = tech["price"]
        setup = scoring.build_trade_setup(price, tech, direction, tech["volatility"]["atr_14"])
        assert setup["valid"]
        assert setup["risk_per_share"] > 0
        # The reward floor must always hold.
        assert setup["risk_reward_t1"] >= MIN_RISK_REWARD - 1e-9
        assert setup["risk_reward_t2"] > setup["risk_reward_t1"]
        if direction == "long":
            assert setup["stop_loss"] < setup["entry_low"] < setup["target_1"] < setup["target_2"]
        else:
            assert setup["stop_loss"] > setup["entry_high"] > setup["target_1"] > setup["target_2"]

    def test_setup_without_atr_is_invalid_not_fatal(self):
        setup = scoring.build_trade_setup(100.0, {"levels": {}}, "long", None)
        assert setup["valid"] is False and "reason" in setup

    def test_event_risks_exclude_data_completeness(self):
        """Data gaps are warnings, never directional counter-evidence."""
        out = scoring.compute(*self._panels())
        assert any("Reduced data completeness" in w for w in out["warnings"])
        assert not any("Reduced data completeness" in r for r in out["event_risks"])


class TestLLMNumberValidation:
    """The guard that makes granting the model numeric authority safe."""

    LONG = {
        "verdict": "BUY", "conviction_pct": 70, "direction": "long",
        "entry_low": 100, "entry_high": 102, "stop_loss": 96,
        "target_1": 115, "target_2": 125,
    }
    SHORT = {
        "verdict": "SELL", "conviction_pct": 60, "direction": "short",
        "entry_low": 100, "entry_high": 102, "stop_loss": 106,
        "target_1": 87, "target_2": 80,
    }

    def test_accepts_sound_plans(self):
        for plan_input in (self.LONG, self.SHORT):
            plan, issues = llm.validate_numbers(plan_input, 100)
            assert issues == []
            assert plan["risk_reward_t1"] >= MIN_RISK_REWARD
            assert plan["author"] == "llm"

    @pytest.mark.parametrize(
        "override,fragment",
        [
            ({"stop_loss": 105}, "not below entry_low"),
            ({"target_1": 104}, "below the"),          # R:R floor
            ({"target_1": 1150}, "away from the price"),  # decimal slip
            ({"verdict": "MAYBE"}, "not one of the five"),
            ({"conviction_pct": 250}, "outside 0-100"),
            ({"direction": "sideways"}, "not long/short"),
            ({"entry_low": 110}, "exceeds entry_high"),
            ({"stop_loss": None}, "missing or non-positive"),
        ],
    )
    def test_rejects_broken_plans(self, override, fragment):
        plan, issues = llm.validate_numbers({**self.LONG, **override}, 100)
        assert plan is None
        assert any(fragment in i for i in issues), issues

    def test_short_stop_must_sit_above_entry(self):
        plan, issues = llm.validate_numbers({**self.SHORT, "stop_loss": 95}, 100)
        assert plan is None
        assert any("not above entry_high" in i for i in issues)

    def test_repair_snaps_target_to_the_floor(self):
        """A sound plan that only misses the R:R floor is repaired, not binned."""
        weak = {**self.LONG, "target_1": 106}  # 0.67:1
        assert llm.validate_numbers(weak, 100)[0] is None  # rejected without repair

        plan, issues = llm.validate_numbers(weak, 100, repair=True)
        assert issues == []
        assert plan["risk_reward_t1"] == pytest.approx(MIN_RISK_REWARD, abs=1e-6)
        # The model's own entry and stop must survive the repair.
        assert plan["entry_low"] == weak["entry_low"]
        assert plan["stop_loss"] == weak["stop_loss"]
        assert plan["adjustments"], "an adjustment must be disclosed"
        assert plan["target_2"] > plan["target_1"]

    def test_repair_never_rescues_a_structurally_broken_plan(self):
        broken = {**self.LONG, "stop_loss": 105}  # stop above entry
        assert llm.validate_numbers(broken, 100, repair=True)[0] is None
