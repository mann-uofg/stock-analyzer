"""Portfolio holdings: import, normalisation, and position analytics.

WHY THERE IS NO BROKER LOGIN
----------------------------
Wealthsimple publishes no public developer API. The only programmatic routes
are unofficial GraphQL clients that require your account password and 2FA code
(breaching their terms, and risking a lock-out), or third-party aggregators
such as SnapTrade and Plaid, which relay your holdings through their servers.
Both conflict with this tool's premise that nothing about your positions leaves
the machine.

So holdings arrive by **file import or direct entry**. Wealthsimple exports
holdings and activity as CSV from the web dashboard; this module reads that,
and most other brokers' exports, by matching columns on meaning rather than on
an exact header. ``Position`` is deliberately broker-agnostic, so a connector
can be added later without touching the analytics.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

# Header synonyms, in priority order. Broker exports disagree on almost every
# column name, so detection matches meaning rather than an exact string.
_COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "stock", "security", "instrument", "name",
               "description", "asset"),
    "quantity": ("quantity", "shares", "units", "qty", "position", "amount held",
                 "number of shares"),
    # Per-share cost only. "Book value" is deliberately absent: in every broker
    # export it is a position TOTAL, and treating it as a unit price inflates
    # the cost basis by the share count.
    "avg_cost": ("average cost", "avg cost", "average price", "avg price",
                 "unit cost", "purchase price", "price paid",
                 "book value per share", "cost basis per share",
                 "average cost per share"),
    "market_price": ("market price", "last price", "current price", "price",
                     "close price", "quote"),
    "market_value": ("market value", "current value", "position value",
                     "total value", "value"),
    "account": ("account name", "account type", "account", "portfolio"),
    "currency": ("market price currency", "currency", "ccy"),
    "exchange": ("exchange", "listing exchange", "market"),
    "mic": ("mic", "mic code", "market identifier code"),
    "direction": ("position direction", "direction", "side", "long short"),
    "security_type": ("security type", "asset type", "instrument type",
                      "product type", "type"),
}

# Crypto is quoted as a PAIR on Yahoo. A bare coin ticker is either a different
# security entirely - "BTC" is the Grayscale Bitcoin Mini Trust ETF, three
# orders of magnitude away from bitcoin - or nothing at all.
_CRYPTO_TYPES = ("crypto", "cryptocurrency", "digital asset", "digital currency",
                 "coin", "virtual currency")

# Fallback for exports with no security-type column. Covers the coins a
# mainstream Canadian broker actually lists.
_KNOWN_CRYPTO = {
    "BTC", "ETH", "DOGE", "SOL", "ADA", "XRP", "LTC", "DOT", "AVAX", "MATIC",
    "SHIB", "LINK", "UNI", "BCH", "XLM", "ETC", "AAVE", "ALGO", "ATOM", "COMP",
    "CRV", "MKR", "SUSHI", "YFI", "BAT", "GRT", "SAND", "MANA", "APE", "FIL",
    "NEAR", "ICP", "HBAR", "VET", "XTZ", "ZEC", "DASH", "QNT", "EGLD", "FTM",
    "ENJ", "CHZ", "CRO", "KNC", "LRC", "SNX", "STORJ", "UMA", "ZRX", "PEPE",
    "SUI", "APT", "ARB", "OP", "TIA", "INJ", "IMX", "STX", "RUNE", "LDO",
}

# Quote currencies Yahoo actually publishes crypto pairs in.
_CRYPTO_QUOTES = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}

# A column is only a per-share cost if it says so. Anything holding "book
# value" without a per-share qualifier is a total.
_TOTAL_ONLY = ("book value", "total cost", "cost basis", "total")
_PER_SHARE = ("per share", "per unit", "unit", "each")

# Exchange and MIC codes to the suffix Yahoo expects. Without this a Canadian
# listing silently resolves to a same-named US security: bare "SHOP" is the
# NYSE line, not the TSX one, at a different price in a different currency.
_EXCHANGE_SUFFIX: dict[str, str] = {
    # Canada
    "TSX": ".TO", "TSE": ".TO", "XTSE": ".TO", "TOR": ".TO",
    "TSXV": ".V", "XTSX": ".V", "VENTURE": ".V",
    "CSE": ".CN", "XCNQ": ".CN", "CNSX": ".CN",
    "NEO": ".NE", "NEOE": ".NE", "AQTS": ".NE",
    # United States - no suffix
    "NASDAQ": "", "XNAS": "", "NSD": "", "NYSE": "", "XNYS": "", "NYS": "",
    "ARCA": "", "ARCX": "", "AMEX": "", "XASE": "", "BATS": "", "BATY": "",
    "OTC": "", "PINK": "",
    # A few common others
    "LSE": ".L", "XLON": ".L",
    "TSXVENTURE": ".V",
    "ASX": ".AX", "XASX": ".AX",
}

# Rows that are not equity positions.
_NON_EQUITY = re.compile(
    r"^(cash|cad|usd|total|net deposits?|interest|dividend|deposit|withdrawal|"
    r"balance|subtotal)\b",
    re.IGNORECASE,
)

# A plausible listed symbol: 1-6 letters, optional exchange/class suffix.
_SYMBOL_RE = re.compile(r"^[A-Z]{1,6}(?:[.\-][A-Z]{1,3})?$")


@dataclass
class Position:
    """One holding, broker-agnostic."""

    symbol: str
    quantity: float
    avg_cost: float | None = None
    account: str | None = None
    currency: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "account": self.account,
            "currency": self.currency,
        }


def _normalise(name: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(name).strip().lower()).strip()


def detect_columns(columns: Iterable[Any]) -> dict[str, str]:
    """Map our canonical field names onto the source frame's column names.

    Exact matches win over substring matches, so a file carrying both
    "Value" and "Market Value" resolves the way a human would read it.
    """
    normalised = {col: _normalise(col) for col in columns}

    # Score every (field, column) pair first, then assign globally by best
    # score. Assigning in field order instead lets a weak partial match claim a
    # column that a later field matches exactly - "Book Value" would be taken
    # by avg_cost (via "book value per share") before book_value ever saw it.
    candidates: list[tuple[int, str, Any]] = []
    for field_name, synonyms in _COLUMN_SYNONYMS.items():
        for col, norm in normalised.items():
            if not norm:
                continue
            # A total masquerading as a unit price is the single most damaging
            # mis-detection here, so per-share fields reject total-only columns
            # outright rather than ranking them low.
            if field_name == "avg_cost" and any(t in norm for t in _TOTAL_ONLY):
                if not any(p in norm for p in _PER_SHARE):
                    continue
            # Currency columns describe a value, they are never the value.
            if field_name in ("market_price", "market_value", "avg_cost") and "currency" in norm:
                continue
            for rank, synonym in enumerate(synonyms):
                if norm == synonym:
                    score = rank                      # exact match
                elif synonym in norm or norm in synonym:
                    score = rank + 100                # partial match, much weaker
                else:
                    continue
                candidates.append((score, field_name, col))
                break

    mapping: dict[str, str] = {}
    taken_columns: set[Any] = set()
    for _, field_name, col in sorted(candidates, key=lambda c: c[0]):
        if field_name in mapping or col in taken_columns:
            continue
        mapping[field_name] = col
        taken_columns.add(col)

    return mapping


def detect_book_value_columns(columns: Iterable[Any]) -> list[tuple[Any, Any | None]]:
    """Find every book-value column, paired with its currency column if present.

    Wealthsimple ships two: ``Book Value (CAD)`` converted to the account's base
    currency, and ``Book Value (Market)`` in the security's own currency. Only
    the latter is comparable with ``Market Price``, so the pairing lets the
    caller choose by currency instead of guessing from the name.
    """
    normalised = {col: _normalise(col) for col in columns}

    values = [
        col for col, norm in normalised.items()
        if any(t in norm for t in ("book value", "cost basis", "total cost"))
        and "currency" not in norm
        and not any(p in norm for p in _PER_SHARE)
    ]

    pairs: list[tuple[Any, Any | None]] = []
    for col in values:
        tokens = set(normalised[col].split())
        wanted = tokens | {"currency"}
        currency_col = next(
            (
                other for other, other_norm in normalised.items()
                if other not in values and set(other_norm.split()) >= wanted
            ),
            None,
        )
        pairs.append((col, currency_col))

    # Prefer market-currency columns; they are the ones that pair with price.
    pairs.sort(key=lambda p: 0 if "market" in normalised[p[0]] else 1)
    return pairs


def is_crypto(symbol: str, security_type: Any = None) -> bool:
    """Whether a holding is a coin rather than a listed security."""
    if security_type is not None:
        text = str(security_type).strip().lower()
        if text and text != "nan":
            return any(t in text for t in _CRYPTO_TYPES)
    return symbol.strip().upper() in _KNOWN_CRYPTO


def resolve_symbol(
    symbol: str,
    exchange: Any = None,
    mic: Any = None,
    security_type: Any = None,
    currency: Any = None,
) -> str:
    """Qualify a raw ticker so Yahoo returns the security actually held.

    Two distinct corrections, both of which otherwise fail silently by
    resolving to a real but *different* instrument:

    * crypto becomes a trading pair (``BTC`` -> ``BTC-CAD``);
    * a non-US listing gains its exchange suffix (``SHOP`` -> ``SHOP.TO``).
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return symbol

    if is_crypto(symbol, security_type):
        if "-" in symbol:
            return symbol  # already a pair
        quote = str(currency or "USD").strip().upper()
        if quote not in _CRYPTO_QUOTES:
            quote = "USD"
        return f"{symbol}-{quote}"

    if "." in symbol:
        return symbol  # already qualified

    for source in (mic, exchange):
        if source is None:
            continue
        key = re.sub(r"[^A-Z]", "", str(source).upper())
        if key in _EXCHANGE_SUFFIX:
            return symbol + _EXCHANGE_SUFFIX[key]
    return symbol


def repair_symbols(positions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Upgrade bare coin tickers in already-saved holdings.

    Positions imported before crypto pairs were handled sit in storage as bare
    tickers, which price against the wrong instrument. This fixes them in place
    so an existing portfolio does not have to be re-imported.
    """
    repaired: list[dict[str, Any]] = []
    changes: list[str] = []
    for position in positions:
        entry = dict(position)
        symbol = str(entry.get("symbol", "")).strip().upper()
        if symbol and "-" not in symbol and symbol in _KNOWN_CRYPTO:
            fixed = resolve_symbol(symbol, security_type="crypto",
                                   currency=entry.get("currency"))
            entry["symbol"] = fixed
            changes.append(f"{symbol}→{fixed}")
        repaired.append(entry)
    return repaired, changes


def _to_number(value: Any) -> float | None:
    """Parse a number out of broker formatting: $1,234.56, (12.00), 1 234,56."""
    if value is None or isinstance(value, bool):
        return None
    # pandas has usually already coerced a numeric column, so overflow arrives
    # here as a float rather than as text.
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if not text:
        return None

    # Try a straight parse before stripping anything: the cleanup below removes
    # the exponent marker, which would silently turn "1e309" into 1309.
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        pass

    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^0-9.,\-]", "", text)
    if not text:
        return None

    # A comma used as the decimal separator (European style).
    if "," in text and "." not in text and re.search(r",\d{1,2}$", text):
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")

    try:
        number = float(text)
    except ValueError:
        return None
    # "1e309" parses to inf, which would silently poison every downstream total
    # with inf/nan rather than failing where it can be seen.
    if not math.isfinite(number):
        return None
    return -number if negative else number


def _extract_symbol(value: Any) -> str | None:
    """Pull a ticker out of a cell that may also carry the company name."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or _NON_EQUITY.match(text):
        return None

    # "NVDA - NVIDIA Corp", "NVIDIA Corp (NVDA)", "NVDA:US", plain "NVDA".
    bracketed = re.search(r"\(([A-Z][A-Z.\-]{0,7})\)", text)
    if bracketed and _SYMBOL_RE.match(bracketed.group(1)):
        return bracketed.group(1)

    # Whole-cell match comes before splitting: class and exchange suffixes are
    # part of the ticker, so splitting on the separator first would turn
    # "BRK-B" into "BRK" and "SHOP.TO" into a different listing entirely.
    whole = text.upper().replace("$", "").strip()
    if _SYMBOL_RE.match(whole):
        return whole

    # For a multi-token cell, only treat the leading token as a ticker if it is
    # already uppercase in the source. "AAPL - Apple Inc." yields AAPL;
    # "Apple Inc." yields nothing, rather than the bogus symbol "APPLE".
    raw_head = re.split(r"[\s\-:|,]+", text)[0].strip().replace("$", "")
    head = raw_head.upper()
    if _SYMBOL_RE.match(head) and raw_head == head:
        return head
    return None


def parse_holdings(source: str | bytes | io.BytesIO | pd.DataFrame) -> tuple[list[Position], list[str]]:
    """Parse a holdings export into positions.

    Returns ``(positions, notes)``. ``notes`` explains what was detected and
    what was skipped, so an unexpected file format is visible rather than
    silently producing an empty portfolio.
    """
    notes: list[str] = []

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        def _rewind() -> None:
            if hasattr(source, "seek"):
                source.seek(0)

        df = None
        try:
            df = pd.read_csv(source)
        except Exception:
            df = None

        # A semicolon- or tab-delimited file parses "successfully" into a single
        # column, so a bare exception check is not enough - the giveaway is one
        # column whose name still contains a delimiter.
        if df is None or (
            df.shape[1] == 1 and re.search(r"[;\t|]", str(df.columns[0]))
        ):
            try:
                _rewind()
                df = pd.read_csv(source, sep=None, engine="python")
            except Exception as exc:
                return [], [f"Could not read the file as CSV ({exc.__class__.__name__})."]

        if df is None:
            return [], ["Could not read the file as CSV."]

    if df.empty:
        return [], ["The file contained no rows."]

    mapping = detect_columns(df.columns)
    if "symbol" not in mapping:
        return [], [
            "No symbol column found. Expected a column named something like "
            f"'Symbol', 'Ticker' or 'Description'. Found: {list(df.columns)[:12]}"
        ]

    book_columns = detect_book_value_columns(df.columns)
    notes.append(
        "Detected columns — "
        + ", ".join(f"{k}: '{v}'" for k, v in mapping.items())
        + (f", book value: {[c for c, _ in book_columns]}" if book_columns else "")
    )

    positions: list[Position] = []
    skipped = 0
    suffixed: list[str] = []

    def _cell(field: str) -> Any:
        return row.get(mapping[field]) if field in mapping else None

    for _, row in df.iterrows():
        raw_symbol = _extract_symbol(_cell("symbol"))
        if not raw_symbol:
            skipped += 1
            continue

        quantity = _to_number(_cell("quantity"))
        avg_cost = _to_number(_cell("avg_cost"))
        market_price = _to_number(_cell("market_price"))
        market_value = _to_number(_cell("market_value"))

        price_currency = _cell("currency")
        price_currency = (
            str(price_currency).strip().upper()
            if price_currency is not None and str(price_currency).strip().lower() != "nan"
            else None
        )

        symbol = resolve_symbol(
            raw_symbol, _cell("exchange"), _cell("mic"),
            _cell("security_type"), price_currency,
        )
        if symbol != raw_symbol:
            suffixed.append(f"{raw_symbol}→{symbol}")

        # Pick the book value quoted in the same currency as the price. Using
        # the account's base-currency column instead would divide a converted
        # total by the share count and report a cost basis in the wrong
        # currency entirely.
        book = None
        for value_col, currency_col in book_columns:
            candidate = _to_number(row.get(value_col))
            if candidate is None:
                continue
            candidate_ccy = (
                str(row.get(currency_col)).strip().upper() if currency_col else None
            )
            if price_currency and candidate_ccy and candidate_ccy == price_currency:
                book = candidate
                break
            if book is None:
                book = candidate

        # Book value is a total, so a per-share cost only comes from dividing.
        if avg_cost is None and book is not None and quantity:
            avg_cost = abs(book / quantity)
        if quantity is None and market_value is not None and market_price:
            quantity = abs(market_value / market_price)
        if quantity is None and market_value is not None and avg_cost:
            quantity = abs(market_value / avg_cost)

        if not quantity or quantity == 0:
            skipped += 1
            continue

        # Short positions carry a negative quantity so the book values correctly.
        direction = str(_cell("direction") or "").strip().lower()
        quantity = -abs(quantity) if direction.startswith("short") else abs(quantity)

        account = _cell("account")
        account = str(account).strip() if account is not None else None

        positions.append(
            Position(
                symbol=symbol,
                quantity=float(quantity),
                avg_cost=avg_cost,
                account=None if account in (None, "nan", "") else account,
                currency=price_currency,
                extras={"market_price": market_price} if market_price else {},
            )
        )

    if suffixed:
        notes.append(
            "Qualified symbols: " + ", ".join(sorted(set(suffixed)))
            + " — without this Yahoo returns a different security, or none."
        )

    # One symbol can appear once per account; combine into a single position
    # with a share-weighted average cost.
    merged: dict[str, Position] = {}
    for pos in positions:
        existing = merged.get(pos.symbol)
        if existing is None:
            merged[pos.symbol] = pos
            continue
        total_qty = existing.quantity + pos.quantity
        if existing.avg_cost is not None and pos.avg_cost is not None and total_qty:
            existing.avg_cost = (
                existing.avg_cost * existing.quantity + pos.avg_cost * pos.quantity
            ) / total_qty
        elif existing.avg_cost is None:
            existing.avg_cost = pos.avg_cost
        existing.quantity = total_qty
        if existing.account and pos.account and pos.account not in existing.account:
            existing.account = f"{existing.account}, {pos.account}"

    final = sorted(merged.values(), key=lambda p: p.symbol)
    notes.append(f"Imported {len(final)} position(s).")
    if skipped:
        notes.append(
            f"Skipped {skipped} row(s) with no recognisable symbol or quantity "
            "(cash lines, totals and headers are expected here)."
        )
    return final, notes


def value_positions(
    positions: list[dict[str, Any]],
    quotes: dict[str, float | None],
    fx: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Attach live prices, market value and unrealised P/L to each position.

    ``fx`` maps a symbol to the multiplier converting its native currency into
    the portfolio's base currency. Without it every figure stays in its own
    currency, and totals across a mixed-currency book would be meaningless -
    adding USD to CAD produces a number that is not money.
    """
    fx = fx or {}
    rows: list[dict[str, Any]] = []
    for pos in positions:
        symbol = str(pos.get("symbol", "")).upper()
        quantity = pos.get("quantity") or 0
        avg_cost = pos.get("avg_cost")
        price = quotes.get(symbol)
        rate = fx.get(symbol, 1.0)

        market_value = price * quantity if price else None
        cost_basis = avg_cost * quantity if avg_cost else None
        pnl = (
            market_value - cost_basis
            if market_value is not None and cost_basis is not None
            else None
        )

        rows.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": avg_cost,
                "price": price,
                "currency": pos.get("currency"),
                "fx_rate": rate,
                # Native currency, as the broker reports it.
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealised_pnl": pnl,
                # Base currency, the only figures safe to add together.
                "market_value_base": market_value * rate if market_value is not None else None,
                "cost_basis_base": cost_basis * rate if cost_basis is not None else None,
                "unrealised_pnl_base": pnl * rate if pnl is not None else None,
                # A percentage is currency-invariant, so it needs no conversion.
                "unrealised_pnl_pct": (
                    pnl / cost_basis * 100 if pnl is not None and cost_basis else None
                ),
                "account": pos.get("account"),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]], betas: dict[str, float | None] | None = None,
              sectors: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Portfolio-level rollup: value, P/L, concentration, weighted beta."""
    betas = betas or {}
    sectors = sectors or {}

    # Totals and weights use base-currency figures throughout; mixing
    # currencies in a sum is the one arithmetic error a portfolio view must
    # never make.
    priced = [r for r in rows if r.get("market_value_base") is not None]
    total_value = sum(r["market_value_base"] for r in priced)
    total_cost = sum(
        r["cost_basis_base"] for r in rows if r.get("cost_basis_base") is not None
    )

    for row in rows:
        row["weight_pct"] = (
            row["market_value_base"] / total_value * 100
            if row.get("market_value_base") is not None and total_value
            else None
        )

    weighted_beta = None
    if total_value:
        covered = 0.0
        acc = 0.0
        for row in priced:
            beta = betas.get(row["symbol"])
            if beta is None:
                continue
            acc += beta * row["market_value"]
            covered += row["market_value"]
        # Normalise over covered value only, so an unpriced or beta-less
        # holding does not drag the figure toward zero.
        if covered:
            weighted_beta = acc / covered

    allocation: dict[str, float] = {}
    for row in priced:
        sector = sectors.get(row["symbol"]) or "Unknown"
        allocation[sector] = allocation.get(sector, 0.0) + row["market_value"]
    if total_value:
        allocation = {
            k: v / total_value * 100
            for k, v in sorted(allocation.items(), key=lambda kv: -kv[1])
        }

    weights = sorted(
        (r["weight_pct"] for r in rows if r.get("weight_pct") is not None), reverse=True
    )
    pnl = total_value - total_cost if total_value and total_cost else None

    return {
        "positions": len(rows),
        "priced_positions": len(priced),
        "total_value": total_value or None,
        "total_cost": total_cost or None,
        "unrealised_pnl": pnl,
        "unrealised_pnl_pct": (pnl / total_cost * 100) if pnl is not None and total_cost else None,
        "weighted_beta": weighted_beta,
        "top_weight_pct": weights[0] if weights else None,
        "top5_weight_pct": sum(weights[:5]) if weights else None,
        # Herfindahl index on portfolio weights: 1/HHI is the "effective number
        # of positions", a truer read of concentration than a simple count.
        "effective_positions": (
            1 / sum((w / 100) ** 2 for w in weights) if weights else None
        ),
        "sector_allocation_pct": allocation,
    }
