"""Central configuration for the stock analyzer.

All tunables live here so the analytical engine stays free of magic numbers.
Secrets are read from the environment (optionally via a local ``.env`` file);
nothing is ever hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, tolerating blanks and junk."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    # --- LLM providers ---
    #
    # Ollama Cloud first, local Ollama second, deterministic narrative last.
    #
    # The local 9B model is what made a laptop run hot: it pins the GPU for
    # minutes per synthesis. Ollama Cloud runs frontier-scale models on their
    # hardware for nothing on the free tier, which is both far smarter and
    # silent. The trade is real and worth stating: with the cloud provider the
    # analysis payload - tickers, positions, prices - is sent to ollama.com.
    # Leave the key unset and nothing leaves the machine.
    ollama_cloud_key: str = field(
        default_factory=lambda: (
            os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_CLOUD_KEY") or ""
        ).strip()
    )
    ollama_cloud_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com").strip()
    )
    # 120B against the 9B that was running locally. Comfortably overkill for
    # this workload, which is the point: the arithmetic and the hedging in a
    # research note are where small models fail.
    ollama_cloud_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b").strip()
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3.5:9b").strip()
    )
    # Reasoning models burn a lot of tokens in <think> blocks for little gain
    # on structured synthesis, so it is off unless explicitly enabled.
    ollama_think: bool = field(
        default_factory=lambda: os.getenv("OLLAMA_THINK", "").strip().lower()
        in {"1", "true", "yes"}
    )
    ollama_num_ctx: int = field(default_factory=lambda: int(_env_float("OLLAMA_NUM_CTX", 16384)))
    # Output budget. Authority mode emits ten extra fields (verdict, conviction,
    # five price levels, direction, rationale) on top of the narrative, and a
    # budget that is too small truncates the JSON mid-string - which surfaces
    # as a parse failure and a silent fall back to engine numbers.
    ollama_num_predict: int = field(
        default_factory=lambda: int(_env_float("OLLAMA_NUM_PREDICT", 3072))
    )

    # --- Behaviour ---
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(_env_float("CACHE_TTL_SECONDS", 900))
    )
    request_timeout: int = field(default_factory=lambda: int(_env_float("REQUEST_TIMEOUT", 20)))
    llm_timeout: int = field(default_factory=lambda: int(_env_float("LLM_TIMEOUT", 180)))
    risk_free_fallback: float = field(
        default_factory=lambda: _env_float("RISK_FREE_FALLBACK", 0.04)
    )


SETTINGS = Settings()

# --- Analytical constants -------------------------------------------------

BENCHMARKS = ("SPY", "QQQ")
RISK_FREE_TICKER = "^IRX"  # 13-week T-bill yield, quoted in percent.

TRADING_DAYS = 252

SMA_PERIODS = (20, 50, 200)
EMA_PERIODS = (20, 50, 200)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
STOCH_PERIOD, STOCH_SMOOTH = 14, 3
CCI_PERIOD = 20
BB_PERIOD, BB_STD = 20, 2
ATR_PERIOD = 14
ADX_PERIOD = 14
VWAP_PERIOD = 14
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLE = 2.0

# Rolling windows for beta/alpha, in trading days.
BETA_WINDOWS = {"1y": 252, "3y": 756}

# Historical-volatility window used as the IV comparison baseline.
HV_WINDOW = 30
IV_LOOKBACK_DAYS = 252

# Options: how far around spot to consider "near the money", as a fraction.
NTM_BAND = 0.10

# Trade construction.
ATR_STOP_MULTIPLE = 1.5
MIN_RISK_REWARD = 2.0
# A structural level further than this many ATRs from entry is ignored when
# placing the stop: honouring it would inflate risk (and therefore the
# reward-target) far beyond what the instrument's daily range justifies.
MAX_STOP_ATR = 2.5

# Weights for the deterministic composite score. Keys map to scoring buckets;
# each bucket emits a score in [-1, 1] which is then weighted and rescaled.
SCORE_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "volatility": 0.10,
    "volume": 0.10,
    "fundamental": 0.15,
    "options": 0.10,
}

VERDICT_BANDS = (
    (75, "STRONG BUY"),
    (60, "BUY"),
    (40, "HOLD"),
    (25, "SELL"),
    (0, "STRONG SELL"),
)
