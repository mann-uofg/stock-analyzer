"""Tiny on-disk cache with TTL.

Network round-trips to Yahoo dominate runtime, and re-running the analyzer on
the same ticker within a few minutes is the common workflow. This keeps repeat
runs near-instant without introducing a real cache dependency.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Callable, TypeVar

from .config import CACHE_DIR, SETTINGS

T = TypeVar("T")


def _key_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{namespace}_{digest}.pkl"


def get_or_fetch(
    namespace: str,
    key: str,
    fetcher: Callable[[], T],
    ttl: int | None = None,
    use_cache: bool = True,
) -> T:
    """Return a cached value, or call ``fetcher`` and cache its result.

    A failed read or a corrupt cache entry is never fatal: we simply refetch.
    """
    ttl = SETTINGS.cache_ttl_seconds if ttl is None else ttl
    path = _key_path(namespace, key)

    if use_cache and ttl > 0 and path.exists():
        try:
            if time.time() - path.stat().st_mtime < ttl:
                with path.open("rb") as fh:
                    return pickle.load(fh)
        except Exception:
            # Corrupt or unreadable entry - fall through and refetch.
            pass

    value = fetcher()

    if use_cache and ttl > 0:
        try:
            with path.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Caching is best-effort; never fail the analysis over it.
            pass

    return value


def clear() -> int:
    """Delete every cached entry. Returns the number of files removed."""
    removed = 0
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
