# Stock Analyzer

A zero-cost equity research terminal — as a hosted web app, a local dashboard,
or a terminal command. Technicals, risk metrics, option Greeks, fundamentals,
chart patterns, ranked news, a scored verdict, and a risk-managed trade plan
written up by an LLM.

---

## Hosting it for free

**1. Get an Ollama Cloud key** (free, no card). Sign in at
[ollama.com](https://ollama.com) → *Settings* → *API keys* → create one.

**2. Deploy** at [share.streamlit.io](https://share.streamlit.io):

- *Create app* → *Deploy a public app from GitHub*
- Repository `mann-uofg/stock-analyzer`, branch `main`, file `app.py`
- Open **Advanced settings → Secrets** before deploying and paste:

```toml
OLLAMA_API_KEY = "your-key-here"
OLLAMA_CLOUD_MODEL = "gpt-oss:120b"
STOCK_ANALYZER_SHARED = "1"
```

That last line is not optional. It switches storage from disk to the browser
session, so holdings are never written to a server other people can reach.

Community Cloud's free tier gives **one private app**, unlimited public ones,
about **1 GB of memory**, and sleeps an app after 12 hours idle — it wakes on
the next visit. There are no custom domains, so the URL is assigned:

<https://mann-uofg-stock-analyzer-app-yndwjm.streamlit.app>

Pushing to `main` redeploys on its own; there is no separate deploy step.

### What this costs you in privacy

With a cloud key set, the analysis payload — tickers, positions, prices — is
sent to ollama.com for the write-up. Market data already comes from Yahoo
either way.

The local-model fallback still works, but only if you have pulled a model:
`ollama pull qwen3.5:9b`. Without one, and without a key, the deterministic
engine still produces the full verdict, scores and trade plan — you lose the
written narrative, nothing else. The numbers were never the model's to begin
with; it explains them, and the validator checks its arithmetic against the
engine either way.

On a shared host your watchlist and holdings live **in the browser session
only**. Use *Save / restore your data* in the sidebar to download them as a
small JSON file and load it back next visit; nothing is stored server-side.

## Running it locally

```bash
streamlit run app.py          # dashboard at localhost:8501
```
```bash
python main.py --ticker NVDA  # terminal dashboard
```

**No subscriptions, no paid tiers.** A cloud key is optional; without one the
app uses a local model, and without that it still produces a complete
deterministic report.

### Picking the cloud model

Ollama Cloud serves a few dozen models and the list moves quickly. Which is
best *here* is not a general-intelligence question — this job needs valid JSON
across nineteen fields and arithmetic that satisfies a 2:1 reward-to-risk
floor, which is exactly where smaller models fail. And because the free tier
meters **GPU time**, the largest model is not automatically the right one for
something you run many times a day.

So measure it rather than guessing:

```bash
python scripts/compare_models.py --ticker NVDA
```

It sends each candidate the same prompt the app sends, on a real payload from
your own data, and runs the answer through the same validator that guards the
live trade plan — reporting valid JSON, field completeness, whether the
arithmetic passed unaided, and latency. Then set the winner:

```bash
OLLAMA_CLOUD_MODEL=<whatever won>
```

**Measured result (2 Sept 2026, free tier):**

| Model | Reachable | Valid JSON | Arithmetic | Time |
|---|---|---|---|---|
| **gpt-oss:120b** | yes | 17/17 | **passed** (R:R 2.18) | **7.6s** |
| nemotron-3-super | yes | truncated | — | 45.5s |
| glm-5.3, glm-5.3-flash | **402 — paid** | — | — | — |
| deepseek-v4-pro / -flash | **402 — paid** | — | — | — |
| kimi-k2.6, qwen3.5:397b | **402 — paid** | — | — | — |

Two things that only measuring reveals: most of the newer, larger models are
**not on the free tier at all** — they return HTTP 402 regardless of how good
they are. And of the two that were reachable, the one that worked did so in
7.6 seconds, roughly twenty times faster than the 9B this replaced.

`nemotron-3-super` is free and failed only by truncating its JSON, so it may
pass with a larger `OLLAMA_NUM_PREDICT`. Worth a retry if you ever want an
alternative.

### Which model runs the write-up

| | Speed | Cost to your machine | Quality |
|---|---|---|---|
| **Ollama Cloud** (`OLLAMA_API_KEY` set) | seconds | none | frontier-scale |
| **Local Ollama** (no key, daemon running) | 1–3 minutes | pins the GPU, runs hot | 9B-class |
| **Deterministic** (neither) | instant | none | no prose, full numbers |

The local path is what made a laptop run hot: a 9B model holds ~6 GB on the
GPU and computes for minutes per note. If you have a cloud key set and want
that memory back:

```bash
brew services stop ollama
```

---

## The dashboard

`streamlit run app.py` opens a local web UI with three pages.

**Research** — one symbol, in depth. An interactive candlestick chart (zoom,
pan, hover) carrying moving averages, the Bollinger envelope, swing
support/resistance and the trade plan drawn on the price axis; then tabs for
the written analysis, technicals and beta table, the option chain with Greeks,
fundamentals and the earnings record, and the raw JSON.

**Watchlist** — symbols you are tracking, ranked by *when* they are
interesting rather than just whether. Two scores per name:

| | Weighted toward | Horizon |
|---|---|---|
| **Near term** | momentum, trend, volume, options positioning | days to weeks |
| **Long term** | fundamentals, primary trend | quarters to years |

A name high on one and low on the other is telling you something a single
score cannot. Valuation carries no weight near-term (it does not move a stock
over a fortnight); today's RSI carries almost none long-term (it is noise at
that horizon).

**Portfolio** — your holdings, valued live, each one analysed exactly like a
watchlist entry, plus a book-level rollup: unrealised P/L, share-weighted
5y-monthly beta, sector mix, and concentration measured as *effective
positions* (1/Herfindahl — well below your position count means the book is
more concentrated than it looks).

Both Watchlist and Portfolio analyse on load and **refresh themselves hourly**
via `st.fragment(run_every=...)`, with a manual refresh available. The written
analysis is on demand only, since local synthesis takes a minute or two.

### Connecting a brokerage account

There is no "Connect Wealthsimple" button, and that is deliberate.
**Wealthsimple publishes no public developer API.** The two available routes
both fail this tool's premise:

* Unofficial GraphQL clients (`ws-api`, `Wsimple`) require your account
  password and 2FA code, breach Wealthsimple's terms, and risk a lock-out.
* Aggregators (SnapTrade, Plaid) work, but relay your holdings through a
  third-party server and need an API key.

So holdings arrive by **CSV import or direct entry**, and never leave the
machine. In Wealthsimple: *Activity → Download CSV*. Columns are matched by
meaning rather than exact header, so most brokers' exports work unchanged —
comma, semicolon or tab delimited; `$1,234.56`, `(12.00)` and `95,50` all
parse; cash lines and totals are skipped; short positions carry a negative
quantity; the same symbol across two accounts merges with a share-weighted
average cost. `Position` is broker-agnostic, so a connector can be added later
without touching the analytics.

Three details in Wealthsimple's export are worth knowing, because each one
silently produces a wrong answer if handled naively:

* **Book value is a position total, not a unit price.** A per-share cost only
  ever comes from dividing it by the quantity, so columns containing "book
  value" are excluded from unit-price detection outright unless they say "per
  share".
* **There are two book-value columns** — `Book Value (CAD)` converted to the
  account's base currency, and `Book Value (Market)` in the security's own
  currency. Only the latter is comparable with `Market Price`. Each is paired
  with its currency column and chosen by matching the price currency; picking
  the wrong one reported a USD holding's cost basis in CAD.
* **`Exchange` / `MIC` decide the ticker.** Bare `SHOP` is the NYSE listing at
  150.03 USD; `SHOP.TO` is the TSX listing at 209.00 CAD. They are different
  securities. The suffix is applied from the export's own exchange column.
* **Crypto must be a pair.** Bare `BTC` is not bitcoin — on Yahoo it is the
  Grayscale Bitcoin Mini Trust ETF, about $28 against bitcoin's five figures,
  and `DOGE` returns nothing at all. `Security Type` drives the conversion to
  `BTC-CAD` / `DOGE-USD`, with a known-coin fallback for exports lacking that
  column. An explicit equity type always wins, since several coin tickers
  collide with real stocks (`MU`, `COMP`, `LINK`, `APE`).

Holdings saved before crypto pairs were handled are repaired in place on load,
so an existing portfolio never has to be re-imported.

**Mixed currencies are converted before anything is summed.** A book holding
both USD and CAD lines has no meaningful total until it is in one currency;
the base is whichever currency holds the most positions, converted at the spot
rate, and the per-position table stays in native currency. Percentage returns
are currency-invariant and left alone. If a rate cannot be fetched the affected
positions are flagged rather than quietly counted at 1:1.

---

## Privacy model

This matters enough to be explicit about.

| What | Where it goes |
|---|---|
| Price/options/fundamentals fetch | Yahoo Finance (unavoidable — it *is* the market data) |
| Your analysis payload, scores, and the LLM reasoning over them | `localhost` only |
| API keys | None exist. There is no hosted-LLM code path to configure. |

There is deliberately **no** Gemini/OpenAI/Groq/OpenRouter integration. The only
model call is a POST to your local Ollama daemon. If Ollama is not running, the
tool falls back to a deterministic narrative rather than reaching for a network
service.

---

## Setup

Requires Python 3.11+ (developed and tested on 3.13).

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

Or with stock tooling:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Local LLM (recommended)

```bash
brew install ollama && brew services start ollama
ollama pull qwen3.5:9b
```

Any instruct-tuned model works — point `OLLAMA_MODEL` at it. A 7–9B model at
Q4 needs roughly 7 GB of free RAM. Synthesis takes about a minute on an M3 Pro
(≈12 s of that is the cold model load; subsequent runs within Ollama's
keep-alive window skip it).

Configuration is optional — copy `.env.example` to `.env` only if you want to
change defaults:

```bash
cp .env.example .env
```

> **Troubleshooting Ollama.** If requests hang and the log shows
> `unknown runner engine, expected --imagegen-engine or --mlx-engine`, a
> Homebrew upgrade has left a stale `ollama serve` running against a newer
> runner binary. Fix with `brew services restart ollama`.

---

## Usage

```bash
python main.py --ticker NVDA                  # full dashboard
python main.py --ticker AAPL,TSLA,AMD         # batch
python main.py --ticker SPY --no-llm          # skip the model, deterministic narrative
python main.py --ticker NVDA --json out.json  # also dump the full payload
python main.py --ticker NVDA --no-options     # faster; skip the chain
python main.py --clear-cache
```

| Flag | Effect |
|---|---|
| `-t, --ticker` | Symbol, or several comma-separated |
| `-p, --period` | History window: `1y`, `2y`, `3y`, `5y` (default), `max` |
| `--no-llm` | Deterministic narrative only |
| `--engine-numbers` | Engine owns the verdict and levels; model only narrates |
| `--no-cache` | Bypass the 15-minute disk cache |
| `--no-options` | Skip option-chain analysis |
| `--json PATH` | Write the complete payload as JSON |
| `-q, --quiet` | No progress spinners |
| `--clear-cache` | Purge cached data and exit |

---

## What it computes

**Technicals** — SMA/EMA 20/50/200 with golden/death-cross detection and
regime state, RSI(14), MACD(12,26,9) with signal and histogram, Stochastic,
CCI(20), Bollinger Bands (upper/lower/%B/bandwidth), ATR(14), ADX with ±DI,
OBV, VWAP(14), 20-day volume-spike detection, and swing-pivot support and
resistance.

**Risk** — Beta and Jensen's alpha against SPY and QQQ over 1y and 3y daily
windows, with R² and a statistical-weakness flag; annualised realised
volatility with its 1-year percentile; Sharpe, Sortino, and max drawdown.

**Derivatives** — near-the-money chain with Black-Scholes delta, gamma, theta
(per day), and vega (per 1% vol); ATM implied volatility; IV/HV ratio; IV rank
and percentile proxies; put/call ratios; gamma concentration and squeeze
detection.

**Fundamentals** — trailing/forward P/E, PEG, P/S, EV/EBITDA, FCF yield,
revenue growth; four quarters of EPS surprise history with beat rate; forward
EPS and revenue consensus; analyst price targets; days to next earnings and the
average absolute historical post-earnings move.

**Verdict** — a weighted composite across six buckets producing `STRONG BUY` →
`STRONG SELL`, a 0–100 conviction score, and an ATR-derived trade setup with
entry range, stop, and targets enforcing a minimum 2:1 reward-to-risk.

---

## Design decisions worth knowing

### The model owns the call, but not unchecked

By default the local model issues the verdict, the conviction score and every
price level. The deterministic engine in `analyzer/scoring.py` scores the same
evidence and is handed to the model as a *reference baseline* it may overrule.

Whatever comes back is validated before you see it, and the two failure classes
are treated differently:

* **Structural** — a verdict outside the five allowed values, a stop on the
  wrong side of the entry, targets out of order, a level more than 50% from
  spot. The plan is incoherent and is rejected outright; the engine's numbers
  are shown instead, with the reason printed.
* **Arithmetic** — the reward-to-risk floor. A 9B model reliably reads
  structure well and then misplaces the target by a few percent. It gets one
  retry with the specific failure fed back; if it still misses, target 1 is
  snapped out to exactly 2:1 while **keeping the model's entry and stop**, and
  the adjustment is disclosed on screen.

Run `--engine-numbers` (or toggle *Model sets the numbers* off) to invert this:
the engine owns every figure and the model only writes the analysis.

Either way, **the tool is fully functional with no LLM installed** — the
deterministic path produces a complete verdict and narrative on its own.

A caveat worth stating: a generative model is not deterministic. Across
repeated runs on identical NVDA data the model returned conviction scores of
85%, 94% and 85%, against the engine's stable 56%. That variance is inherent to
granting it numeric authority, and it is why both figures are always shown side
by side.

### Implied volatility is solved, not trusted

Yahoo's free option endpoint has degraded. For many symbols it returns
`openInterest = 0`, `bid = ask = 0`, and a quantized placeholder
`impliedVolatility` — values like `0.00001`, `0.03126`, `0.12501`, which are
exact binary fractions rather than market volatility.

So the analyzer inverts Black-Scholes with Brent's method to recover IV from
the traded price. Validating Yahoo's quote by repricing it does *not* work: for
deep in- or out-of-the-money strikes vega is tiny, so even a placeholder
volatility reprices within tolerance while implying badly wrong Greeks. Solving
from price is used wherever a price exists, and the quote is only a fallback.

On live NVDA data this moves ATM IV from a nonsensical 1.56% to 41.25% against
30-day realised volatility of 39.79% — and the resulting surface satisfies
put-call parity, with same-strike call and put deltas summing to 1.0.

### Metrics that cannot be computed honestly are reported as unavailable

Free data has no historical IV surface, so a true 52-week IV Rank does not
exist here; what is shown is explicitly a proxy ranked against the realised-vol
distribution. Yahoo exposes no historical revenue *estimates*, so revenue
surprise history is omitted rather than fabricated — actuals and YoY growth
appear instead. When open interest is missing, put/call and gamma fall back to
volume weighting and say so. A missing value renders as a dim `n/a`, never a
zero.

### Three betas, and one of them matches the websites

A beta depends entirely on two choices: the lookback and the sampling
frequency. Yahoo, Google and most vendors publish **5 years of monthly**
returns. This tool reports that figure *and* 1y/3y **daily** windows, which
react far faster to a change in regime.

The 5y-monthly column is marked ★ so you can cross-check it against any public
source. Measured against Yahoo's published beta:

| | KO | BRK-B | AAPL | NVDA | TSLA | JNJ |
|---|---|---|---|---|---|---|
| This tool (5y monthly) | 0.333 | 0.610 | 1.078 | 2.210 | 1.818 | 0.239 |
| Yahoo published | 0.342 | 0.607 | 1.086 | 2.215 | 1.827 | 0.231 |

Every one agrees within ~1%. The daily windows will look different — for the
same names above, KO's 1y daily beta is −0.28 — and that is the estimator
working, not failing: a defensive staple genuinely decouples from the index
over a single year. Where R² < 0.10 the row is flagged ⚠, because a beta
regressed off near-zero correlation carries little information however precise
it looks.

### `ta`, not `pandas-ta`

`pandas-ta` pins `numpy<2.3` and silently downgrades the environment on
install. `ta` is pure pandas/numpy with no transitive pins and covers every
indicator in the spec.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

129 offline tests, no network. They cover put-call parity, delta parity,
Black-Scholes cross-validated against `py_vollib` to 1e-9, IV solver
round-trips, the placeholder-IV rejection path (including the deep-ITM case
that defeats naive validation), beta recovery on synthetic levered series, the
low-R² flag, trade-setup invariants (R:R floor, stop/entry/target ordering),
the LLM number validator in both directions plus its repair path, holdings
import against the real Wealthsimple header (book value never read as a unit
price, market-currency column preferred, exchange suffixes, crypto pairs and
their collisions with equity tickers, short positions) plus delimiter sniffing,
currency and accounting formats, class suffixes like `BRK-B`, duplicate merging
and column-detection conflicts; mixed-currency totals; horizon-score weighting
and renormalisation; persistence round-trips; and graceful degradation on short
or empty history.

`pyflakes` is worth running alongside the suite — the tests exercise
`analyzer/`, not the Streamlit views, so an undefined name in a view only
surfaces at runtime:

```bash
.venv/bin/python -m pyflakes analyzer/*.py views/*.py app.py main.py
```

---

## Project layout

```
analyzer/
  config.py        settings, weights, thresholds
  cache.py         TTL disk cache
  datafeed.py      yfinance access with retry/backoff
  indicators.py    technical indicators
  risk.py          beta (daily + monthly), alpha, volatility, drawdown
  options.py       chain analysis, Greeks, IV solving
  fundamentals.py  valuation, earnings, consensus
  scoring.py       composite verdict + trade construction
  llm.py           local Ollama synthesis, number validation, fallback
  prompts.py       system prompts and output schemas
  horizon.py       near-term vs long-term reweighting
  portfolio.py     holdings import, valuation, concentration
  store.py         local JSON persistence
  charts.py        Plotly charts for the dashboard
  report.py        rich terminal rendering
  engine.py        pipeline orchestration
views/
  research.py      single-symbol deep dive
  watchlist.py     tracked symbols, ranked by horizon
  portfolio.py     holdings import and book analytics
  common.py        shared formatting and cached analysis
app.py             Streamlit shell and navigation
main.py            typer CLI
data/              watchlist.json, portfolio.json (local, git-ignorable)
```

---

## Limitations

- Yahoo data is delayed and occasionally degraded; the tool reports quality
  caveats but cannot repair a missing feed.
- Open interest is currently unavailable from the free endpoint for many
  symbols, so put/call and gamma metrics fall back to volume.
- Greeks assume European exercise and no dividend yield. For American options
  on dividend-paying names, early-exercise effects are not modelled.
- The composite weights are a reasonable prior, not a backtested edge. Nothing
  here has been validated against forward returns.

Analytical output only. Not investment advice.
