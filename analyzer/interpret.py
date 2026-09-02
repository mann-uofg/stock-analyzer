"""Turn raw indicator values into something a non-specialist can act on.

Every number this app computes is standard and correct, and almost none of it
is self-explanatory. "Bollinger %B 0.94" is meaningless unless you already know
the scale runs 0 to 1 and that the top of it means something. A reader who has
to look up the scale before they can use the figure is not being helped by it.

So each metric here carries three things the bare number does not: which band
it falls in, whether that band is encouraging or worrying, and one sentence on
what it implies for a decision. The number itself is never replaced - the point
is to make it legible, not to hide it.

Two deliberate limits:

Bands are conventional, not universal. A software company on 30x earnings is
ordinary and a utility on 30x is dear, so valuation readings say out loud that
the sector matters rather than pretending a single threshold settles it.

Nothing here is advice. A reading explains what a measurement means; whether to
buy is the verdict engine's job, and ultimately the reader's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Tones map onto the palette the rest of the UI already uses.
GOOD, WARN, BAD, NEUTRAL = "good", "warn", "bad", "neutral"


@dataclass(frozen=True)
class Band:
    """One region of a metric's scale, and what being in it means."""
    upto: float          # inclusive upper edge; the last band uses infinity
    verdict: str         # the short label a reader sees first
    tone: str
    plain: str           # one sentence, may reference {value}


@dataclass(frozen=True)
class Reading:
    """A metric, resolved against its bands."""
    key: str
    label: str
    value: float | None
    display: str
    verdict: str
    tone: str
    plain: str
    position: float | None       # 0..1 along the scale, for the marker
    scale: tuple[float, float]
    # (start, end, tone) spans covering 0..1, so the track can be coloured to
    # show what is good and bad *before* the reader looks at the marker.
    zones: list[tuple[float, float, str]] = field(default_factory=list)
    note: str = ""
    higher_is_better: bool | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class Spec:
    label: str
    lo: float
    hi: float
    bands: list[Band]
    digits: int = 1
    suffix: str = ""
    signed: bool = False
    note: str = ""
    higher_is_better: bool | None = None
    what: str = ""       # plain-language gloss of the metric itself


# --- the registry ---------------------------------------------------------
#
# Thresholds follow the conventional readings used across technical analysis
# and equity screening. Where a convention is genuinely contested, the note
# says so rather than presenting one school as settled fact.

SPECS: dict[str, Spec] = {
    # --- momentum ---------------------------------------------------------
    "rsi_14": Spec(
        label="RSI (14)", lo=0, hi=100, digits=1,
        what="How hard the stock has been bought or sold over the last 14 days, "
             "on a 0-100 scale.",
        bands=[
            Band(30, "Oversold", WARN,
                 "Sellers have been in control. Bounces often begin near here, "
                 "but something falling can stay oversold for weeks."),
            Band(45, "Soft", NEUTRAL,
                 "Mild selling pressure. No strong signal either way."),
            Band(55, "Neutral", NEUTRAL,
                 "Buyers and sellers are evenly matched. The chart is not "
                 "arguing for a decision."),
            Band(70, "Strong", GOOD,
                 "Steady buying without being stretched — usually the healthiest "
                 "reading on this scale."),
            Band(float("inf"), "Overbought", WARN,
                 "Buyers have pushed hard. Pullbacks are common from here, "
                 "though strong stocks can stay overbought through a whole run."),
        ],
    ),
    "stoch_k": Spec(
        label="Stochastic %K", lo=0, hi=100, digits=1,
        what="Where today's close sits inside the recent high-low range.",
        bands=[
            Band(20, "Oversold", WARN,
                 "Closing near the bottom of its recent range."),
            Band(80, "Mid-range", NEUTRAL,
                 "Closing in the middle of its recent range — unremarkable."),
            Band(float("inf"), "Overbought", WARN,
                 "Closing near the top of its recent range. Stretched, "
                 "not necessarily wrong."),
        ],
    ),
    "cci_20": Spec(
        label="CCI (20)", lo=-250, hi=250, digits=0, signed=True,
        what="How far price has strayed from its own 20-day average, "
             "measured in typical deviations.",
        bands=[
            Band(-100, "Stretched low", WARN,
                 "Unusually far below its own average."),
            Band(100, "Normal range", NEUTRAL,
                 "Trading close to its own average — nothing unusual."),
            Band(float("inf"), "Stretched high", WARN,
                 "Unusually far above its own average. Often precedes a pause."),
        ],
    ),

    # --- trend ------------------------------------------------------------
    "adx_14": Spec(
        label="ADX (14)", lo=0, hi=60, digits=1,
        note="ADX measures how strong a trend is, never which way it points. "
             "A high reading in a downtrend is a strong downtrend.",
        what="How decisively the stock is trending, in either direction.",
        bands=[
            Band(20, "No real trend", NEUTRAL,
                 "Drifting sideways. Trend-following signals are unreliable here "
                 "and false starts are common."),
            Band(25, "Trend forming", NEUTRAL,
                 "A direction is starting to establish itself, but it is not "
                 "yet convincing."),
            Band(50, "Strong trend", GOOD,
                 "A genuine trend is in place. Whichever way it points, it has "
                 "conviction behind it."),
            Band(float("inf"), "Very strong trend", WARN,
                 "An unusually powerful move. These rarely sustain at this "
                 "intensity and often cool off."),
        ],
    ),
    "bb_percent_b": Spec(
        label="Bollinger %B", lo=-0.2, hi=1.2, digits=2,
        what="Where price sits between the Bollinger bands: 0 is the lower "
             "band, 1 is the upper band.",
        bands=[
            Band(0.0, "Below the lower band", WARN,
                 "Trading below its normal range entirely — a sign of real "
                 "selling pressure."),
            Band(0.2, "Near the lower band", WARN,
                 "Hugging the bottom of its normal range."),
            Band(0.8, "Mid-range", NEUTRAL,
                 "Comfortably inside its normal trading range."),
            Band(1.0, "Near the upper band", WARN,
                 "Hugging the top of its normal range — extended, but that is "
                 "what a strong trend looks like."),
            Band(float("inf"), "Above the upper band", WARN,
                 "Trading above its normal range entirely. Either exceptional "
                 "strength or an overshoot."),
        ],
    ),

    # --- volatility -------------------------------------------------------
    "atr_percent": Spec(
        label="Daily swing (ATR)", lo=0, hi=8, digits=2, suffix="%",
        what="How much this typically moves in a single day, as a percentage "
             "of its price.",
        bands=[
            Band(1.0, "Very calm", GOOD,
                 "Moves less than 1% on a typical day. Easy to hold without "
                 "losing sleep."),
            Band(2.0, "Normal", NEUTRAL,
                 "Ordinary daily movement for a listed stock."),
            Band(4.0, "Lively", WARN,
                 "Swings of this size mean a position needs room to breathe, "
                 "and a tight stop will be taken out by noise alone."),
            Band(float("inf"), "Wild", BAD,
                 "Very large daily swings. Position size matters far more than "
                 "entry price at this level of movement."),
        ],
    ),
    "hv_percentile_1y": Spec(
        label="Volatility vs its own year", lo=0, hi=100, digits=0, suffix="%",
        what="How today's turbulence compares with this stock's own past year.",
        bands=[
            Band(25, "Unusually calm", GOOD,
                 "Quieter than most of the past year. Calm periods do not last "
                 "indefinitely."),
            Band(75, "Typical", NEUTRAL,
                 "About as volatile as it usually is."),
            Band(float("inf"), "Unusually turbulent", WARN,
                 "Choppier than most of the past year — something has the "
                 "market's attention."),
        ],
    ),
    "volume_ratio": Spec(
        label="Volume vs normal", lo=0, hi=4, digits=2, suffix="x",
        what="Today's trading volume against its 20-day average.",
        bands=[
            Band(0.6, "Very quiet", NEUTRAL,
                 "Few shares changing hands. Moves on thin volume carry less "
                 "conviction."),
            Band(1.5, "Normal", NEUTRAL,
                 "Ordinary participation."),
            Band(2.0, "Heavy", GOOD,
                 "Noticeably more interest than usual."),
            Band(float("inf"), "Volume spike", GOOD,
                 "Twice its usual volume or more. Something has happened; a "
                 "price move on volume like this is far more meaningful than "
                 "one without it."),
        ],
    ),

    # --- risk -------------------------------------------------------------
    "beta": Spec(
        label="Beta", lo=0, hi=2.5, digits=2,
        what="How much this moves when the market moves.",
        bands=[
            Band(0.5, "Very defensive", GOOD,
                 "Barely follows the market. For every 1% the market moves, "
                 "this has historically moved about {value}%."),
            Band(0.8, "Defensive", GOOD,
                 "Moves less than the market. For every 1% the market moves, "
                 "this has historically moved about {value}%."),
            Band(1.2, "Moves with the market", NEUTRAL,
                 "Tracks the market closely — roughly {value}% for every 1% "
                 "the market moves."),
            Band(1.5, "Aggressive", WARN,
                 "Amplifies the market. A 1% market move has historically "
                 "meant about {value}% here — in both directions."),
            Band(float("inf"), "Very aggressive", BAD,
                 "Strongly amplifies the market: about {value}% for every 1% "
                 "move. Gains and losses both arrive magnified."),
        ],
    ),
    "r_squared": Spec(
        label="R² (beta reliability)", lo=0, hi=1, digits=2,
        note="This is the number that tells you whether to trust the beta "
             "above it.",
        what="How much of this stock's movement the market actually explains.",
        bands=[
            Band(0.10, "Beta is unreliable", BAD,
                 "The market explains almost none of this stock's movement, so "
                 "the beta figure carries little meaning here."),
            Band(0.30, "Weak relationship", WARN,
                 "The market explains only a small part of what this does. "
                 "Treat the beta as a rough hint."),
            Band(0.60, "Moderate relationship", NEUTRAL,
                 "A fair amount of this stock's movement follows the market."),
            Band(float("inf"), "Strong relationship", GOOD,
                 "This largely moves with the market, so the beta above is "
                 "dependable."),
        ],
    ),
    "sharpe": Spec(
        label="Sharpe ratio", lo=-1, hi=3, digits=2, signed=True,
        what="Return earned for each unit of risk taken — the classic "
             "reward-for-risk score.",
        bands=[
            Band(0.0, "Losing money for the risk", BAD,
                 "Has not rewarded the risk taken. Cash would have done better "
                 "on this measure."),
            Band(0.5, "Poor", WARN,
                 "The returns have not really justified the bumpiness."),
            Band(1.0, "Fair", NEUTRAL,
                 "A reasonable return for the risk carried."),
            Band(2.0, "Good", GOOD,
                 "Solid reward for the risk taken."),
            Band(float("inf"), "Excellent", GOOD,
                 "Unusually strong reward for the risk — rare, and worth "
                 "checking it is not a short lucky run."),
        ],
    ),
    "sortino": Spec(
        label="Sortino ratio", lo=-1, hi=4, digits=2, signed=True,
        note="Like Sharpe, but it only counts downward moves as risk — upside "
             "volatility is not something you need protecting from.",
        what="Reward measured against downside risk alone.",
        bands=[
            Band(0.0, "Losing money for the risk", BAD,
                 "Downside has not been compensated by return."),
            Band(0.7, "Poor", WARN,
                 "Weak reward for the losses endured along the way."),
            Band(1.5, "Fair", NEUTRAL,
                 "Reasonable compensation for the downside risk."),
            Band(2.5, "Good", GOOD, "Strong reward relative to its declines."),
            Band(float("inf"), "Excellent", GOOD,
                 "Exceptional reward for the downside taken on."),
        ],
    ),
    "max_drawdown_1y_pct": Spec(
        label="Worst fall this year", lo=-60, hi=0, digits=1, suffix="%",
        what="The deepest peak-to-trough drop over the past year — what "
             "holding it actually felt like at the worst moment.",
        bands=[
            Band(-35, "Brutal", BAD,
                 "A fall of this size tests anyone's conviction. Be honest "
                 "about whether you would have held through it."),
            Band(-20, "Severe", WARN,
                 "A serious drop. Position size is what makes this survivable."),
            Band(-10, "Normal", NEUTRAL,
                 "An ordinary pullback for an individual stock."),
            Band(float("inf"), "Mild", GOOD,
                 "Has not fallen far from its highs this year."),
        ],
    ),

    # --- options ----------------------------------------------------------
    "iv_hv_ratio": Spec(
        label="Options priced vs reality", lo=0.5, hi=2.0, digits=2, suffix="x",
        note="Above 1 means options are pricing in more movement than the "
             "stock has actually been delivering.",
        what="What options traders expect, against what the stock has really "
             "been doing.",
        bands=[
            Band(0.9, "Options look cheap", GOOD,
                 "The market expects less movement than this stock has actually "
                 "delivered — historically a decent time to buy options rather "
                 "than sell them."),
            Band(1.1, "Fairly priced", NEUTRAL,
                 "Expected movement roughly matches recent reality."),
            Band(1.3, "Options look rich", WARN,
                 "Options are pricing in more drama than the stock has been "
                 "delivering. Often a sign of an event ahead."),
            Band(float("inf"), "Options look expensive", WARN,
                 "A large premium for expected movement — usually earnings or "
                 "news is expected. Buying options here needs a big move just "
                 "to break even."),
        ],
    ),
    "put_call_ratio": Spec(
        label="Put/call ratio", lo=0, hi=2.0, digits=2,
        note="Contrarians read extremes the other way: everyone already "
             "bearish can mean the selling is done.",
        what="Whether traders are buying more downside protection or more "
             "upside bets.",
        bands=[
            Band(0.7, "Bullish positioning", GOOD,
                 "Far more call buying than put buying — traders are leaning "
                 "optimistic."),
            Band(1.0, "Balanced", NEUTRAL,
                 "No strong tilt either way in the options market."),
            Band(1.3, "Cautious", WARN,
                 "More protection being bought than upside — a defensive tilt."),
            Band(float("inf"), "Bearish positioning", BAD,
                 "Heavy demand for downside protection. Traders are worried "
                 "about something."),
        ],
    ),

    # --- valuation --------------------------------------------------------
    # Every band here carries the same caveat, because a single P/E threshold
    # across all industries is the most common way to misread a stock.
    "trailing_pe": Spec(
        label="P/E (trailing)", lo=0, hi=60, digits=1,
        note="Heavily sector-dependent. Fast-growing software routinely trades "
             "where a utility would look absurd — compare with its peers, not "
             "with the market.",
        what="What you pay for each pound of the profit it already earns.",
        bands=[
            Band(10, "Very cheap", GOOD,
                 "Priced low against current earnings. Sometimes a bargain, "
                 "sometimes the market expecting profits to fall."),
            Band(18, "Cheap", GOOD,
                 "Modestly priced against what it currently earns."),
            Band(28, "Fair", NEUTRAL,
                 "Around the long-run market average."),
            Band(45, "Rich", WARN,
                 "Priced for meaningful growth. That growth now has to arrive."),
            Band(float("inf"), "Expensive", BAD,
                 "Priced for exceptional growth. Disappointment tends to be "
                 "punished hard at this level."),
        ],
    ),
    "forward_pe": Spec(
        label="P/E (forward)", lo=0, hi=60, digits=1,
        note="Based on analyst forecasts, which are routinely too optimistic. "
             "Sector-dependent in the same way as the trailing figure.",
        what="What you pay for each pound of profit it is *expected* to earn "
             "next year.",
        bands=[
            Band(10, "Very cheap", GOOD,
                 "Low against expected earnings — if the forecasts hold."),
            Band(18, "Cheap", GOOD, "Modestly priced against expected earnings."),
            Band(28, "Fair", NEUTRAL, "Around the market's usual level."),
            Band(45, "Rich", WARN,
                 "Expensive even against next year's hoped-for profits."),
            Band(float("inf"), "Expensive", BAD,
                 "Very demanding even on optimistic forecasts."),
        ],
    ),
    "peg": Spec(
        label="PEG ratio", lo=0, hi=4, digits=2,
        note="Only meaningful when growth is positive and reasonably steady.",
        what="The P/E measured against the growth rate — the classic test of "
             "whether a high price is justified by fast growth.",
        bands=[
            Band(0.8, "Cheap for its growth", GOOD,
                 "Growth is not fully reflected in the price. Under 1 is the "
                 "traditional bargain marker."),
            Band(1.5, "Fairly priced for its growth", NEUTRAL,
                 "Price and growth are roughly in balance."),
            Band(2.5, "Rich for its growth", WARN,
                 "Paying up well beyond the growth on offer."),
            Band(float("inf"), "Expensive for its growth", BAD,
                 "The price implies far more growth than is currently "
                 "forecast."),
        ],
    ),
    "ev_ebitda": Spec(
        label="EV/EBITDA", lo=0, hi=30, digits=1,
        note="Sector-dependent, but harder to distort than P/E because it "
             "accounts for debt.",
        what="The whole business value — debt included — against its "
             "operating earnings.",
        bands=[
            Band(8, "Cheap", GOOD, "Inexpensive against its operating profits."),
            Band(14, "Fair", NEUTRAL, "A normal multiple for a healthy business."),
            Band(22, "Rich", WARN, "Priced well above its operating earnings."),
            Band(float("inf"), "Expensive", BAD,
                 "A demanding multiple that leaves little room for a stumble."),
        ],
    ),
    "fcf_yield_pct": Spec(
        label="Free cash flow yield", lo=-5, hi=12, digits=2, suffix="%",
        signed=True,
        what="The actual cash the business throws off each year, as a "
             "percentage of its market value.",
        bands=[
            Band(0, "Burning cash", BAD,
                 "Spending more cash than it generates. Sustainable only while "
                 "someone keeps funding it."),
            Band(2, "Thin", WARN,
                 "Generates little spare cash relative to its price."),
            Band(5, "Healthy", GOOD,
                 "Produces solid cash for what you pay — think of it as the "
                 "rate the business itself pays you."),
            Band(float("inf"), "Strong", GOOD,
                 "Throws off a lot of cash for its price. Worth checking it is "
                 "durable rather than a one-off."),
        ],
    ),
    "days_to_earnings": Spec(
        label="Days until earnings", lo=0, hi=90, digits=0,
        note="Earnings routinely move a stock 5-10% in a day regardless of "
             "what any chart says beforehand.",
        what="How long until the next results announcement.",
        bands=[
            Band(7, "Imminent", BAD,
                 "Results land within a week. Technical setups are frequently "
                 "overwhelmed by the reaction — many people simply wait."),
            Band(21, "Approaching", WARN,
                 "Results are close enough to matter for a short-term trade."),
            Band(float("inf"), "Clear for now", GOOD,
                 "No results imminent, so the chart has room to play out."),
        ],
    ),
}

# Metrics where a lower number is the better outcome, used when phrasing
# comparisons.
_LOWER_IS_BETTER = {
    "trailing_pe", "forward_pe", "peg", "ev_ebitda", "put_call_ratio",
    "iv_hv_ratio", "atr_percent",
}


def _format(value: float, spec: Spec) -> str:
    text = f"{value:+.{spec.digits}f}" if spec.signed else f"{value:.{spec.digits}f}"
    return f"{text}{spec.suffix}"


def _band_for(value: float, spec: Spec) -> Band:
    for band in spec.bands:
        if value <= band.upto:
            return band
    return spec.bands[-1]


def _position(value: float, spec: Spec) -> float:
    span = spec.hi - spec.lo
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - spec.lo) / span))


def read(key: str, value: Any) -> Reading | None:
    """Resolve one metric against its bands.

    Returns ``None`` for an unknown metric so a caller can fall back to the
    plain number rather than inventing an interpretation for it.
    """
    spec = SPECS.get(key)
    if spec is None:
        return None

    if value is None:
        return Reading(
            key=key, label=spec.label, value=None, display="—",
            verdict="Not available", tone=NEUTRAL,
            plain="No data for this measure.", position=None,
            scale=(spec.lo, spec.hi), note=spec.note,
            higher_is_better=key not in _LOWER_IS_BETTER,
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    # An infinite or NaN ratio is a division artefact, not a measurement.
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return Reading(
            key=key, label=spec.label, value=None, display="—",
            verdict="Not meaningful", tone=NEUTRAL,
            plain="This figure could not be computed sensibly for this stock.",
            position=None, scale=(spec.lo, spec.hi), note=spec.note,
            higher_is_better=key not in _LOWER_IS_BETTER,
        )

    band = _band_for(numeric, spec)
    display = _format(numeric, spec)
    # Beta's sentence quotes the value back, which is what makes it click:
    # "for every 1% the market moves, this moved 1.7%".
    plain = band.plain.format(value=f"{abs(numeric):.{spec.digits}f}")

    return Reading(
        key=key, label=spec.label, value=numeric, display=display,
        verdict=band.verdict, tone=band.tone, plain=plain,
        position=_position(numeric, spec), scale=(spec.lo, spec.hi),
        zones=zones(key), note=spec.note,
        higher_is_better=key not in _LOWER_IS_BETTER,
    )


def zones(key: str) -> list[tuple[float, float, str]]:
    """The coloured spans of a metric's scale, as fractions from 0 to 1.

    Contiguous by construction: each band runs from where the previous one
    ended, so the track paints without gaps whatever the thresholds are.
    """
    spec = SPECS.get(key)
    if spec is None:
        return []
    out: list[tuple[float, float, str]] = []
    start = 0.0
    for band in spec.bands:
        end = 1.0 if band.upto == float("inf") else _position(band.upto, spec)
        if end > start:
            out.append((start, end, band.tone))
            start = end
    if out and out[-1][1] < 1.0:
        last = out[-1]
        out[-1] = (last[0], 1.0, last[2])
    return out


def what_is(key: str) -> str:
    """The plain-language gloss of a metric, for a help tooltip."""
    spec = SPECS.get(key)
    return spec.what if spec else ""


def read_many(pairs: list[tuple[str, Any]]) -> list[Reading]:
    """Resolve several metrics, dropping any the registry does not know."""
    out = []
    for key, value in pairs:
        reading = read(key, value)
        if reading is not None:
            out.append(reading)
    return out


def concerns(readings: list[Reading]) -> list[Reading]:
    """The readings a reader should look at first, worst news at the top."""
    order = {BAD: 0, WARN: 1, NEUTRAL: 2, GOOD: 3}
    flagged = [r for r in readings if r.ok and r.tone in (BAD, WARN)]
    return sorted(flagged, key=lambda r: order.get(r.tone, 9))


def encouraging(readings: list[Reading]) -> list[Reading]:
    return [r for r in readings if r.ok and r.tone == GOOD]


def summarise(readings: list[Reading], subject: str = "this stock") -> str:
    """One sentence over a group of readings, for the top of a panel."""
    usable = [r for r in readings if r.ok]
    if not usable:
        return "Not enough data to read these measures."

    good = len(encouraging(usable))
    bad = len([r for r in usable if r.tone == BAD])
    warn = len([r for r in usable if r.tone == WARN])
    total = len(usable)

    if bad == 0 and warn == 0:
        return (f"All {total} measures here read positively or neutrally for "
                f"{subject} — nothing on this panel is flashing a warning.")
    if good > (bad + warn):
        return (f"{good} of {total} measures look encouraging, with "
                f"{bad + warn} worth a closer look below.")
    if (bad + warn) > good:
        return (f"{bad + warn} of {total} measures here are cautionary against "
                f"{good} encouraging — the concerns are listed first.")
    return (f"Mixed: {good} encouraging and {bad + warn} cautionary out of "
            f"{total} measures.")
