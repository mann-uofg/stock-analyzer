"""Offline tests for holdings import, horizon scoring, and persistence."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from analyzer import benchmark, horizon, portfolio, review, riskmodel, sizing, store
from analyzer.portfolio import (
    _extract_symbol,
    _to_number,
    detect_book_value_columns,
    detect_columns,
    resolve_symbol,
)

# The real Wealthsimple holdings export header.
WEALTHSIMPLE_HEADER = (
    "Account Name,Account Type,Account Classification,Account Number,Symbol,"
    "Exchange,MIC,Name,Security Type,Quantity,Position Direction,Market Price,"
    "Market Price Currency,Book Value (CAD),Book Value Currency (CAD),"
    "Book Value (Market),Book Value Currency (Market),Market Value,"
    "Market Value Currency,Market Unrealized Returns,"
    "Market Unrealized Returns Currency\n"
)


class TestWealthsimpleExport:
    """The real export format, which broke three separate assumptions."""

    def _file(self, *rows: str) -> io.BytesIO:
        return io.BytesIO((WEALTHSIMPLE_HEADER + "".join(rows)).encode())

    US_ROW = ("TFSA,TFSA,Registered,ABC,NVDA,NASDAQ,XNAS,NVIDIA Corporation,Equity,"
              "25,long,217.50,USD,6200.00,CAD,4512.50,USD,5437.50,USD,925.00,USD\n")
    CA_ROW = ("TFSA,TFSA,Registered,ABC,SHOP,TSX,XTSE,Shopify Inc.,Equity,"
              "15,long,95.50,CAD,1200.00,CAD,1200.00,CAD,1432.50,CAD,232.50,CAD\n")

    def test_book_value_is_never_treated_as_a_unit_price(self):
        """Regression: book value is a position total, not cost per share."""
        positions, _ = portfolio.parse_holdings(self._file(self.US_ROW))
        # 4512.50 USD / 25 shares, NOT the raw 4512.50 and NOT the CAD column.
        assert positions[0].avg_cost == pytest.approx(180.50)

    def test_book_value_matches_the_price_currency(self):
        """The CAD column would divide a converted total by the share count."""
        positions, _ = portfolio.parse_holdings(self._file(self.US_ROW))
        assert positions[0].currency == "USD"
        # 6200 CAD / 25 = 248.00 would be the wrong answer.
        assert positions[0].avg_cost != pytest.approx(248.00)

    def test_avg_cost_never_maps_to_a_book_value_column(self):
        mapping = detect_columns(WEALTHSIMPLE_HEADER.strip().split(","))
        assert "book value" not in str(mapping.get("avg_cost", "")).lower()
        assert mapping["market_price"] == "Market Price"
        assert mapping["quantity"] == "Quantity"
        assert mapping["symbol"] == "Symbol"

    def test_market_currency_book_column_is_preferred(self):
        pairs = detect_book_value_columns(WEALTHSIMPLE_HEADER.strip().split(","))
        assert pairs[0] == ("Book Value (Market)", "Book Value Currency (Market)")
        assert ("Book Value (CAD)", "Book Value Currency (CAD)") in pairs

    def test_canadian_listing_gets_its_suffix(self):
        """Bare SHOP is the NYSE line at a different price in a different currency."""
        positions, notes = portfolio.parse_holdings(self._file(self.CA_ROW))
        assert positions[0].symbol == "SHOP.TO"
        assert any("SHOP→SHOP.TO" in n for n in notes)

    def test_us_listing_is_left_alone(self):
        positions, _ = portfolio.parse_holdings(self._file(self.US_ROW))
        assert positions[0].symbol == "NVDA"

    def test_market_price_is_captured(self):
        positions, _ = portfolio.parse_holdings(self._file(self.US_ROW))
        assert positions[0].extras["market_price"] == pytest.approx(217.50)

    def test_short_position_is_negative(self):
        short = self.US_ROW.replace(",long,", ",short,")
        positions, _ = portfolio.parse_holdings(self._file(short))
        assert positions[0].quantity == -25


class TestCryptoSymbols:
    """A bare coin ticker resolves to an unrelated security, not to nothing.

    "BTC" on Yahoo is the Grayscale Bitcoin Mini Trust ETF — around $28 against
    bitcoin's five figures — so this fails silently rather than loudly.
    """

    @pytest.mark.parametrize(
        "symbol,security_type,currency,expected",
        [
            ("BTC", "Cryptocurrency", "CAD", "BTC-CAD"),
            ("BTC", "Cryptocurrency", "USD", "BTC-USD"),
            ("DOGE", "Crypto", None, "DOGE-USD"),
            ("ETH", None, "CAD", "ETH-CAD"),          # via the known-coin list
            ("BTC-USD", "crypto", "CAD", "BTC-USD"),  # already a pair
            ("SOL", "Cryptocurrency", "XYZ", "SOL-USD"),  # unsupported quote
        ],
    )
    def test_pairs(self, symbol, security_type, currency, expected):
        assert resolve_symbol(symbol, None, None, security_type, currency) == expected

    @pytest.mark.parametrize("symbol", ["MU", "COMP", "LINK", "APE"])
    def test_explicit_equity_type_wins_over_the_coin_list(self, symbol):
        """Several coin tickers collide with real equities."""
        assert resolve_symbol(symbol, None, None, "Equity", "USD") == symbol

    def test_crypto_row_in_a_wealthsimple_export(self):
        header = WEALTHSIMPLE_HEADER
        row = ("Crypto,CRYPTO,Non-Registered,ABC,BTC,,,Bitcoin,Cryptocurrency,"
               "0.5,long,88000.00,CAD,40000.00,CAD,44000.00,CAD,44000.00,CAD,"
               "4000.00,CAD\n")
        positions, notes = portfolio.parse_holdings(
            io.BytesIO((header + row).encode())
        )
        assert positions[0].symbol == "BTC-CAD"
        assert positions[0].avg_cost == pytest.approx(88000.0)  # 44000 / 0.5
        assert any("BTC→BTC-CAD" in n for n in notes)

    def test_repair_upgrades_already_saved_holdings(self):
        saved = [
            {"symbol": "BTC", "quantity": 0.5, "avg_cost": 1.0, "currency": "CAD"},
            {"symbol": "MU", "quantity": 10, "avg_cost": 100.0, "currency": "USD"},
        ]
        repaired, changes = portfolio.repair_symbols(saved)
        assert [p["symbol"] for p in repaired] == ["BTC-CAD", "MU"]
        assert changes == ["BTC→BTC-CAD"]

    def test_repair_is_idempotent(self):
        once, _ = portfolio.repair_symbols(
            [{"symbol": "BTC", "quantity": 1, "currency": "CAD"}]
        )
        twice, changes = portfolio.repair_symbols(once)
        assert [p["symbol"] for p in twice] == ["BTC-CAD"]
        assert changes == []


class TestExchangeSuffix:
    @pytest.mark.parametrize(
        "symbol,exchange,mic,expected",
        [
            ("SHOP", "TSX", "XTSE", "SHOP.TO"),
            ("XEQT", "TSX", "XTSE", "XEQT.TO"),
            ("NVDA", "NASDAQ", "XNAS", "NVDA"),
            ("AAPL", "NYSE", "XNYS", "AAPL"),
            ("ABC", "TSXV", "XTSX", "ABC.V"),
            ("DEF", "CSE", "XCNQ", "DEF.CN"),
            ("GHI", "NEO", "NEOE", "GHI.NE"),
            ("SHOP.TO", "TSX", "XTSE", "SHOP.TO"),   # already qualified
            ("XYZ", "UNKNOWN", None, "XYZ"),          # unknown venue left alone
            ("XYZ", None, None, "XYZ"),
        ],
    )
    def test_suffixes(self, symbol, exchange, mic, expected):
        assert resolve_symbol(symbol, exchange, mic) == expected


class TestSymbolExtraction:
    @pytest.mark.parametrize(
        "cell,expected",
        [
            ("NVDA", "NVDA"),
            ("nvda", "NVDA"),             # a bare token in a symbol column is a ticker
            ("BRK-B", "BRK-B"),           # class suffix survives
            ("SHOP.TO", "SHOP.TO"),       # exchange suffix survives
            ("AAPL - Apple Inc.", "AAPL"),
            ("NVIDIA Corp (NVDA)", "NVDA"),
            ("NVDA:US", "NVDA"),
            ("$AMD", "AMD"),
            ("Apple Inc.", None),         # company name only
            ("Tesla Motors", None),
            ("Cash", None),
            ("CASH", None),
            ("Total", None),
            ("Net deposits", None),
            ("", None),
            (None, None),
        ],
    )
    def test_extraction(self, cell, expected):
        assert _extract_symbol(cell) == expected

    def test_class_suffix_is_not_truncated(self):
        """Regression: splitting on '-' first turned BRK-B into BRK."""
        assert _extract_symbol("BRK-B") == "BRK-B"


class TestNumberParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234.56", 1234.56),
            ("$1,234.56", 1234.56),
            ("(12.00)", -12.0),          # accounting negative
            ("1 234.56", 1234.56),
            ("95,50", 95.5),             # comma decimal separator
            ("12,345", 12345.0),         # comma thousands separator
            ("", None),
            ("n/a", None),
            (None, None),
            (42, 42.0),
            # Scientific notation must survive the punctuation cleanup, which
            # would otherwise strip the exponent and read 1e309 as 1309.
            ("1.5e3", 1500.0),
            ("1e308", 1e308),
            # Non-finite values are rejected rather than poisoning totals.
            ("1e309", None),
            ("-1e309", None),
            ("inf", None),
            ("nan", None),
            (float("inf"), None),
            (float("nan"), None),
            (True, None),
        ],
    )
    def test_parsing(self, raw, expected):
        assert _to_number(raw) == expected

    def test_overflow_row_is_dropped_not_imported_as_infinity(self):
        positions, _ = portfolio.parse_holdings(
            io.BytesIO(b"Symbol,Quantity,Average Cost\nNVDA,1e309,1e309\nAAPL,10,100\n")
        )
        assert [p.symbol for p in positions] == ["AAPL"]


class TestColumnDetection:
    def test_exact_match_beats_substring(self):
        mapping = detect_columns(["Value", "Market Value", "Symbol", "Quantity"])
        assert mapping["market_value"] == "Market Value"
        assert mapping["symbol"] == "Symbol"

    def test_synonyms(self):
        mapping = detect_columns(["ticker", "shares", "avg price", "account type"])
        assert mapping["symbol"] == "ticker"
        assert mapping["quantity"] == "shares"
        assert mapping["avg_cost"] == "avg price"
        assert mapping["account"] == "account type"

    def test_no_column_claimed_twice(self):
        mapping = detect_columns(["Symbol", "Quantity"])
        assert len(set(mapping.values())) == len(mapping)


class TestHoldingsImport:
    def _csv(self, text: str) -> io.BytesIO:
        return io.BytesIO(text.encode())

    def test_wealthsimple_style_export(self):
        positions, notes = portfolio.parse_holdings(self._csv(
            "Symbol,Quantity,Average Cost,Book Value,Market Value,Account\n"
            "NVDA,25,180.50,4512.50,5437.50,TFSA\n"
            'AAPL,10,"$212.00","2,120.00","2,700.00",TFSA\n'
            "Cash,,,,2500.00,TFSA\n"
            "Total,,,16562.50,21137.50,\n"
        ))
        assert [p.symbol for p in positions] == ["AAPL", "NVDA"]
        assert positions[1].quantity == 25
        assert positions[0].avg_cost == pytest.approx(212.0)
        assert any("Skipped" in n for n in notes)

    def test_semicolon_delimited_file(self):
        """Regression: a non-comma delimiter parses into one column silently."""
        positions, _ = portfolio.parse_holdings(self._csv(
            "ticker;shares;avg price\nSHOP.TO;15;95,50\nBRK-B;3;410,25\n"
        ))
        assert {p.symbol for p in positions} == {"SHOP.TO", "BRK-B"}

    def test_avg_cost_derived_from_book_value(self):
        positions, _ = portfolio.parse_holdings(self._csv(
            "Symbol,Quantity,Book Value\nMSFT,10,4000.00\n"
        ))
        assert positions[0].avg_cost == pytest.approx(400.0)

    def test_duplicate_symbols_merge_with_weighted_cost(self):
        positions, _ = portfolio.parse_holdings(self._csv(
            "Symbol,Quantity,Average Cost,Account\n"
            "NVDA,10,100.00,TFSA\n"
            "NVDA,30,200.00,RRSP\n"
        ))
        assert len(positions) == 1
        assert positions[0].quantity == 40
        # (10*100 + 30*200) / 40
        assert positions[0].avg_cost == pytest.approx(175.0)

    def test_missing_symbol_column_is_reported(self):
        positions, notes = portfolio.parse_holdings(self._csv("foo,bar\n1,2\n"))
        assert positions == []
        assert any("No symbol column" in n for n in notes)

    def test_empty_file(self):
        positions, notes = portfolio.parse_holdings(pd.DataFrame())
        assert positions == []
        assert notes


class TestValuation:
    POSITIONS = [
        {"symbol": "NVDA", "quantity": 25, "avg_cost": 180.0},
        {"symbol": "AAPL", "quantity": 10, "avg_cost": 200.0},
    ]

    def test_pnl_maths(self):
        rows = portfolio.value_positions(self.POSITIONS, {"NVDA": 200.0, "AAPL": 180.0})
        nvda, aapl = rows[0], rows[1]
        assert nvda["market_value"] == 5000
        assert nvda["unrealised_pnl"] == pytest.approx(500)
        assert nvda["unrealised_pnl_pct"] == pytest.approx(11.111, abs=1e-3)
        assert aapl["unrealised_pnl"] == pytest.approx(-200)

    def test_missing_price_does_not_crash(self):
        rows = portfolio.value_positions(self.POSITIONS, {"NVDA": None, "AAPL": 180.0})
        assert rows[0]["market_value"] is None
        assert rows[0]["unrealised_pnl"] is None

    def test_weighted_beta_normalises_over_covered_value(self):
        """A holding with no beta must not drag the average toward zero."""
        rows = portfolio.value_positions(self.POSITIONS, {"NVDA": 200.0, "AAPL": 180.0})
        summary = portfolio.summarise(rows, betas={"NVDA": 2.0, "AAPL": None})
        assert summary["weighted_beta"] == pytest.approx(2.0)

    def test_mixed_currency_totals_use_the_base_currency(self):
        """Adding USD to CAD produces a number that is not money."""
        positions = [
            {"symbol": "NVDA", "quantity": 10, "avg_cost": 100.0, "currency": "USD"},
            {"symbol": "SHOP.TO", "quantity": 10, "avg_cost": 50.0, "currency": "CAD"},
        ]
        rows = portfolio.value_positions(
            positions, {"NVDA": 200.0, "SHOP.TO": 100.0},
            fx={"NVDA": 1.4, "SHOP.TO": 1.0},
        )
        assert rows[0]["market_value"] == 2000          # native USD
        assert rows[0]["market_value_base"] == 2800     # converted to CAD
        assert rows[1]["market_value_base"] == 1000

        summary = portfolio.summarise(rows)
        assert summary["total_value"] == 3800           # not the naive 3000
        # Weights must also be computed on converted values.
        assert rows[0]["weight_pct"] == pytest.approx(2800 / 3800 * 100)

    def test_pnl_percent_is_currency_invariant(self):
        rows = portfolio.value_positions(
            [{"symbol": "X", "quantity": 1, "avg_cost": 100.0, "currency": "USD"}],
            {"X": 150.0}, fx={"X": 1.4},
        )
        assert rows[0]["unrealised_pnl_pct"] == pytest.approx(50.0)
        assert rows[0]["unrealised_pnl"] == pytest.approx(50.0)
        assert rows[0]["unrealised_pnl_base"] == pytest.approx(70.0)

    def test_absent_fx_leaves_values_unchanged(self):
        rows = portfolio.value_positions(self.POSITIONS, {"NVDA": 200.0, "AAPL": 180.0})
        assert rows[0]["market_value_base"] == rows[0]["market_value"]

    def test_concentration(self):
        rows = portfolio.value_positions(
            [{"symbol": "A", "quantity": 1, "avg_cost": 1}], {"A": 100.0}
        )
        summary = portfolio.summarise(rows)
        assert summary["top_weight_pct"] == pytest.approx(100)
        assert summary["effective_positions"] == pytest.approx(1.0)


class TestHorizonScoring:
    def _payload(self, trend, momentum, fundamental):
        return {
            "verdict": {"buckets": {
                "trend": {"score": trend}, "momentum": {"score": momentum},
                "fundamental": {"score": fundamental},
                "volume": {"score": 0.0}, "volatility": {"score": 0.0},
                "options": {"score": 0.0},
            }},
            "fundamental": {"earnings": {"days_to_earnings": 40}},
        }

    def test_momentum_drives_near_term_not_long_term(self):
        out = horizon.compute(self._payload(0.0, 1.0, 0.0))
        assert out["near_term_score"] > out["long_term_score"]
        assert out["bias"] == "near term"

    def test_fundamentals_drive_long_term_not_near_term(self):
        out = horizon.compute(self._payload(0.0, 0.0, 1.0))
        assert out["long_term_score"] > out["near_term_score"]
        assert out["bias"] == "long term"

    def test_missing_bucket_renormalises(self):
        """Absent options data must not drag every score toward neutral."""
        payload = {"verdict": {"buckets": {
            "trend": {"score": 1.0}, "momentum": {"score": 1.0},
            "volume": {"score": 1.0}, "volatility": {"score": 1.0},
            "fundamental": {"score": 1.0},
        }}}
        out = horizon.compute(payload)
        assert out["near_term_score"] == pytest.approx(100.0)
        assert out["long_term_score"] == pytest.approx(100.0)

    def test_no_buckets_yields_none(self):
        out = horizon.compute({"verdict": {"buckets": {}}})
        assert out["near_term_score"] is None
        assert out["bias"] is None

    def test_imminent_earnings_called_out(self):
        payload = self._payload(0.5, 0.5, 0.5)
        payload["fundamental"]["earnings"]["days_to_earnings"] = 5
        assert "Earnings in 5 days" in horizon.compute(payload)["summary"]

    def test_rank_puts_unscored_last(self):
        rows = [{"symbol": "A", "near_term_score": None},
                {"symbol": "B", "near_term_score": 10.0},
                {"symbol": "C", "near_term_score": 90.0}]
        assert [r["symbol"] for r in horizon.rank(rows, "near_term")] == ["C", "B", "A"]


class TestStore:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "WATCHLIST_FILE", tmp_path / "w.json")
        monkeypatch.setattr(store, "PORTFOLIO_FILE", tmp_path / "p.json")

    def test_watchlist_round_trip(self):
        assert store.load_watchlist() == []
        assert store.add_to_watchlist("nvda", "note")[0] is True
        assert store.watchlist_symbols() == ["NVDA"]

    def test_duplicates_rejected(self):
        store.add_to_watchlist("NVDA")
        changed, message = store.add_to_watchlist("nvda")
        assert changed is False and "already" in message

    def test_blank_symbol_rejected(self):
        assert store.add_to_watchlist("   ")[0] is False

    def test_remove(self):
        store.add_to_watchlist("NVDA")
        store.remove_from_watchlist("nvda")
        assert store.watchlist_symbols() == []

    def test_corrupt_file_returns_empty(self):
        store.WATCHLIST_FILE.write_text("{not json")
        assert store.load_watchlist() == []

    def test_portfolio_round_trip(self):
        store.save_portfolio([{"symbol": "NVDA", "quantity": 5, "avg_cost": 100.0}])
        assert store.load_portfolio()[0]["symbol"] == "NVDA"
        assert store.last_updated(store.PORTFOLIO_FILE)


class TestBookReview:
    """Findings about the portfolio as a whole."""

    def _rows(self, *specs):
        rows = []
        for symbol, value, pnl_pct, currency in specs:
            rows.append({
                "symbol": symbol, "market_value_base": value,
                "market_value": value, "unrealised_pnl_pct": pnl_pct,
                "currency": currency, "price": 1.0,
                "weight_pct": None,
            })
        total = sum(r["market_value_base"] for r in rows)
        for r in rows:
            r["weight_pct"] = r["market_value_base"] / total * 100
        return rows

    def test_flags_single_name_concentration(self):
        rows = self._rows(("A", 800, 0, "USD"), ("B", 200, 0, "USD"))
        out = review.compute(rows, {"top_weight_pct": 80, "positions": 2}, {})
        assert any("80% of the book" in f["headline"] for f in out["findings"])
        assert out["findings"][0]["level"] == "critical"

    def test_flags_sector_concentration(self):
        rows = self._rows(("A", 500, 0, "USD"), ("B", 500, 0, "USD"))
        out = review.compute(
            rows, {"sector_allocation_pct": {"Technology": 65.0, "Utilities": 35.0}}, {}
        )
        assert any("Technology is 65%" in f["headline"] for f in out["findings"])

    def test_unknown_sector_is_not_flagged(self):
        """Crypto has no sector; that is missing data, not concentration."""
        rows = self._rows(("A", 1000, 0, "USD"))
        out = review.compute(rows, {"sector_allocation_pct": {"Unknown": 100.0}}, {})
        assert not any("Unknown is" in f["headline"] for f in out["findings"])

    def test_flags_currency_exposure(self):
        rows = self._rows(("A", 600, 0, "USD"), ("B", 400, 0, "CAD"))
        out = review.compute(rows, {}, {}, base_currency="CAD")
        assert any("in USD" in f["headline"] for f in out["findings"])

    def test_unpriced_holding_is_critical(self):
        rows = self._rows(("A", 100, 0, "USD"))
        rows[0]["price"] = None
        out = review.compute(rows, {}, {})
        assert out["findings"][0]["level"] == "critical"
        assert "could not be priced" in out["findings"][0]["headline"]

    def test_weighted_scores_favour_larger_positions(self):
        rows = self._rows(("BIG", 900, 0, "USD"), ("SMALL", 100, 0, "USD"))
        analysed = {
            "BIG": {"near_term_score": 90.0, "long_term_score": 90.0},
            "SMALL": {"near_term_score": 10.0, "long_term_score": 10.0},
        }
        out = review.compute(rows, {}, analysed)
        assert out["near_term_score"] == pytest.approx(82.0)

    def test_quiet_book_says_so(self):
        rows = self._rows(("A", 500, 2, "USD"), ("B", 500, 3, "USD"))
        out = review.compute(rows, {"positions": 2, "effective_positions": 2.0}, {})
        assert out["findings"][0]["level"] == "good"

    def test_findings_are_severity_ordered(self):
        rows = self._rows(("A", 900, -30, "USD"), ("B", 100, -30, "USD"))
        rows[1]["price"] = None
        out = review.compute(
            rows, {"top_weight_pct": 90, "positions": 2, "effective_positions": 1.2}, {}
        )
        levels = [review.SEVERITY_ORDER[f["level"]] for f in out["findings"]]
        assert levels == sorted(levels)


class TestPositionSizing:
    """Fixed-fractional sizing: the stop distance sets the share count."""

    def test_risk_budget_is_respected(self):
        plan = sizing.size_position(10000, 2.0, 100.0, 90.0)
        assert plan["valid"]
        assert plan["dollars_at_risk"] == pytest.approx(200.0)
        assert plan["shares"] == pytest.approx(20.0)
        assert plan["bound_by"] == "risk budget"

    def test_wider_stop_buys_fewer_shares(self):
        # Cap lifted so the risk budget alone governs, which is what this
        # asserts; with the default cap the tight stop is capped instead and
        # its risk is deliberately lower.
        tight = sizing.size_position(10000, 2.0, 100.0, 95.0, max_position_pct=100)
        wide = sizing.size_position(10000, 2.0, 100.0, 80.0, max_position_pct=100)
        assert wide["shares"] < tight["shares"]
        # Risk stays constant - that is the whole point of the method.
        assert wide["dollars_at_risk"] == pytest.approx(tight["dollars_at_risk"])

    def test_position_cap_lowers_risk_below_the_budget(self):
        """When the cap binds, the trade risks less than the budget allows."""
        capped = sizing.size_position(10000, 2.0, 100.0, 99.0, max_position_pct=20)
        assert capped["position_pct"] == pytest.approx(20.0)
        assert capped["dollars_at_risk"] < capped["risk_budget"]

    def test_position_cap_binds_when_stop_is_very_tight(self):
        plan = sizing.size_position(10000, 2.0, 100.0, 99.9, max_position_pct=20)
        assert plan["position_pct"] == pytest.approx(20.0)
        assert "cap" in plan["bound_by"]

    def test_short_setup_uses_entry_low(self):
        setup = {"valid": True, "direction": "short", "entry_low": 100.0,
                 "entry_high": 102.0, "stop_loss": 106.0, "target_1": 90.0,
                 "target_2": 85.0}
        plan = sizing.sizing_for_setup(setup, 10000, 2.0)
        assert plan["entry_used"] == 100.0
        assert plan["risk_per_share"] == pytest.approx(6.0)

    def test_long_setup_uses_the_least_favourable_fill(self):
        setup = {"valid": True, "direction": "long", "entry_low": 100.0,
                 "entry_high": 102.0, "stop_loss": 96.0, "target_1": 115.0,
                 "target_2": 125.0}
        plan = sizing.sizing_for_setup(setup, 10000, 2.0)
        assert plan["entry_used"] == 102.0
        # Reported profit is rounded to cents, so compare at that tolerance.
        assert plan["profit_at_target_1"] == pytest.approx(
            (115.0 - 102.0) * plan["shares"], abs=0.01
        )

    @pytest.mark.parametrize("account,risk,entry,stop", [
        (0, 2.0, 100.0, 90.0),        # no account size
        (10000, 2.0, 100.0, 100.0),   # stop equals entry
        (10000, 0, 100.0, 90.0),      # no risk budget
        (10000, 2.0, -5.0, 90.0),     # nonsense entry
    ])
    def test_degenerate_inputs_are_rejected(self, account, risk, entry, stop):
        plan = sizing.size_position(account, risk, entry, stop)
        assert not plan["valid"]
        assert plan["warnings"]

    def test_whole_shares_too_expensive_explains_itself(self):
        plan = sizing.size_position(500, 1.0, 900.0, 850.0, allow_fractional=False)
        assert not plan["valid"]
        assert "fractional" in plan["warnings"][0]

    def test_reckless_risk_is_flagged(self):
        plan = sizing.size_position(10000, 8.0, 100.0, 90.0)
        assert plan["valid"]
        assert any("halve the account" in w for w in plan["warnings"])


class TestRiskModel:
    def _hist(self, values):
        idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
        return pd.DataFrame({"Close": values}, index=idx)

    def test_perfectly_correlated_names_cluster(self):
        import numpy as np
        base = np.linspace(100, 200, 300)
        histories = {
            "A": self._hist(base),
            "B": self._hist(base * 2),
            "C": self._hist(base[::-1]),
        }
        corr = riskmodel.correlation_matrix(histories)
        groups = riskmodel.clusters(corr)
        assert any({"A", "B"} <= set(g) for g in groups)

    def test_stress_scales_by_portfolio_beta(self):
        rows = [{"symbol": "A", "market_value_base": 1000.0}]
        out = riskmodel.stress(rows, {"A": 2.0})
        drop = next(s for s in out["scenarios"] if s["market_move_pct"] == -10)
        assert drop["portfolio_move_pct"] == pytest.approx(-20.0)
        assert drop["resulting_value"] == pytest.approx(800.0)
        assert out["portfolio_beta"] == pytest.approx(2.0)

    def test_stress_without_betas_is_empty_not_wrong(self):
        rows = [{"symbol": "A", "market_value_base": 1000.0}]
        out = riskmodel.stress(rows, {"A": None})
        assert out["scenarios"] == []
        assert out["portfolio_beta"] is None


class TestBenchmarkComparison:
    def _hist(self, start, end, n=300):
        import numpy as np
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": np.linspace(start, end, n)}, index=idx)

    def test_weighted_return_favours_larger_positions(self):
        rows = [
            {"symbol": "BIG", "market_value_base": 900.0},
            {"symbol": "SMALL", "market_value_base": 100.0},
        ]
        histories = {"BIG": self._hist(100, 200), "SMALL": self._hist(100, 100)}
        result, coverage = benchmark.weighted_return(rows, histories, 252)
        assert coverage == pytest.approx(1.0)
        assert result is not None and result > 40  # dominated by BIG

    def test_coverage_reported_when_history_is_missing(self):
        rows = [
            {"symbol": "A", "market_value_base": 500.0},
            {"symbol": "B", "market_value_base": 500.0},
        ]
        histories = {"A": self._hist(100, 150)}
        _, coverage = benchmark.weighted_return(rows, histories, 252)
        assert coverage == pytest.approx(0.5)

    def test_divergence_between_holdings_and_actual_pnl_is_called_out(self):
        rows = [{"symbol": "A", "market_value_base": 1000.0}]
        histories = {"A": self._hist(100, 300)}
        out = benchmark.compare(rows, histories, {}, actual_pnl_pct=-20.0)
        assert any("entry price and timing" in c for c in out["commentary"])


class TestSharedHostStorage:
    """On a shared server, one visitor's holdings must never reach another."""

    def test_local_host_uses_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "WATCHLIST_FILE", tmp_path / "w.json")
        monkeypatch.setattr(store, "is_shared_host", lambda: False)
        store.save_watchlist([{"symbol": "NVDA"}])
        assert (tmp_path / "w.json").exists()
        assert store.watchlist_symbols() == ["NVDA"]

    def test_shared_host_never_writes_to_disk(self, tmp_path, monkeypatch):
        """Regression: writing holdings to a shared filesystem publishes them."""
        target = tmp_path / "p.json"
        monkeypatch.setattr(store, "PORTFOLIO_FILE", target)
        monkeypatch.setattr(store, "is_shared_host", lambda: True)
        monkeypatch.setattr(store, "_session_state", lambda: {})
        store.save_portfolio([{"symbol": "NVDA", "quantity": 25}])
        assert not target.exists(), "positions must not touch a shared disk"

    def test_shared_host_round_trips_through_the_session(self, tmp_path, monkeypatch):
        session: dict = {}
        monkeypatch.setattr(store, "PORTFOLIO_FILE", tmp_path / "p.json")
        monkeypatch.setattr(store, "is_shared_host", lambda: True)
        monkeypatch.setattr(store, "_session_state", lambda: session)
        store.save_portfolio([{"symbol": "MU", "quantity": 3}])
        assert store.load_portfolio()[0]["symbol"] == "MU"
        # A different visitor is a different session dict, and sees nothing.
        monkeypatch.setattr(store, "_session_state", lambda: {})
        assert store.load_portfolio() == []

    def test_shared_host_without_a_session_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "PORTFOLIO_FILE", tmp_path / "p.json")
        monkeypatch.setattr(store, "is_shared_host", lambda: True)
        monkeypatch.setattr(store, "_session_state", lambda: None)
        assert store.load_portfolio() == []

    def test_export_import_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "WATCHLIST_FILE", tmp_path / "w.json")
        monkeypatch.setattr(store, "PORTFOLIO_FILE", tmp_path / "p.json")
        monkeypatch.setattr(store, "SETTINGS_FILE", tmp_path / "s.json")
        monkeypatch.setattr(store, "is_shared_host", lambda: False)
        store.add_to_watchlist("NVDA", "note")
        store.save_portfolio([{"symbol": "MU", "quantity": 2, "avg_cost": 100.0}])
        bundle = store.export_state()

        store.save_watchlist([])
        store.clear_portfolio()
        notes = store.import_state(bundle)
        assert store.watchlist_symbols() == ["NVDA"]
        assert store.load_portfolio()[0]["symbol"] == "MU"
        assert any("position" in n for n in notes)

    def test_import_rejects_rubbish(self):
        assert "not a saved state" in store.import_state("nonsense")[0]
