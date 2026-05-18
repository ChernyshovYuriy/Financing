"""
value_universe.py — Filter Canadian ticker universe for value screening.

Applies a deterministic filter pipeline to reduce ~4,700 Canadian tickers
to a sub-list of quality, liquid, profitable equities suitable for
downstream value factor screening.

Run from repo root:
    python py/value_universe.py
    python py/value_universe.py --limit 50          # test on first 50 tickers
    python py/value_universe.py --no-cache          # skip cache, fresh data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/can_tickers_full"
DEFAULT_OUTPUT = "data/can_tickers_value_universe"
DEFAULT_CACHE_DIR = ".cache/yfinance"
DEFAULT_LOG_DIR = "logs"
CACHE_TTL_HOURS = 24


# ─── Logging ─────────────────────────────────────────────────────────────────


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("value_universe")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir / "value_universe_errors.log", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    logger.addHandler(fh)
    return logger


# ─── I/O ─────────────────────────────────────────────────────────────────────


def filter_exchange(tickers: list[str], include_venture: bool) -> list[str]:
    """Drop TSX Venture (.V) tickers unless include_venture is True."""
    if include_venture:
        return list(tickers)
    return [t for t in tickers if not t.endswith(".V")]


def load_tickers(path: Path) -> list[str]:
    """Read tickers from file, strip blanks/comments, deduplicate (order-preserving)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    raw = [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]
    seen: set[str] = set()
    unique: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def save_tickers(tickers: list[str], path: Path) -> None:
    """Write tickers to file, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(tickers) + "\n") if tickers else "", encoding="utf-8")


# ─── Cache ────────────────────────────────────────────────────────────────────


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return cache_dir / f"{safe}.json"


def _load_cache(ticker: str, cache_dir: Path) -> Optional[dict]:
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched_at > timedelta(hours=CACHE_TTL_HOURS):
            return None
        return data
    except Exception:
        return None


def _save_cache(ticker: str, data: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe(v: object) -> object:
        return None if isinstance(v, float) and math.isnan(v) else v

    _cache_path(ticker, cache_dir).write_text(
        json.dumps({k: _safe(v) for k, v in data.items()}, indent=2),
        encoding="utf-8",
    )


# ─── Fetch ────────────────────────────────────────────────────────────────────


def fetch_metrics(
        ticker: str,
        *,
        use_cache: bool,
        cache_dir: Path,
        sleep_seconds: float,
        logger: logging.Logger,
) -> Optional[dict]:
    """
    Fetch fundamental metrics for one ticker.

    Returned dict fields:
        ticker, fetched_at, market_cap, quote_type, trailing_pe,
        book_value, total_revenue, trailing_eps, dollar_vol_30d

    Returns None on unrecoverable failure (logged to errors log).
    Sleeps only when an actual network request is made.
    """
    if use_cache:
        hit = _load_cache(ticker, cache_dir)
        if hit is not None:
            return hit

    try:
        t = yf.Ticker(ticker)
        info: dict = t.info or {}

        dollar_vol_30d: Optional[float] = None
        try:
            hist = t.history(period="2mo", auto_adjust=True)
            if not hist.empty:
                recent = hist.tail(30)
                dv = (recent["Close"] * recent["Volume"]).mean()
                if not pd.isna(dv):
                    dollar_vol_30d = float(dv)
        except Exception as exc:
            logger.warning(f"{ticker}: history fetch failed — {exc}")

        metrics: dict = {
            "ticker": ticker,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "market_cap": info.get("marketCap"),
            "quote_type": info.get("quoteType"),
            "trailing_pe": info.get("trailingPE"),
            "book_value": info.get("bookValue"),
            "total_revenue": info.get("totalRevenue"),
            "trailing_eps": info.get("trailingEps"),
            "dollar_vol_30d": dollar_vol_30d,
        }

        if use_cache:
            _save_cache(ticker, metrics, cache_dir)

        time.sleep(sleep_seconds)
        return metrics

    except Exception as exc:
        logger.error(f"{ticker}: fetch failed — {exc}")
        return None


# ─── Filters ─────────────────────────────────────────────────────────────────


def apply_filters(
        tickers: list[str],
        *,
        min_market_cap: float,
        min_dollar_volume: float,
        include_unprofitable: bool,
        use_cache: bool,
        cache_dir: Path,
        sleep_seconds: float,
        logger: logging.Logger,
) -> tuple[list[str], dict[str, int]]:
    """
    Apply filters 2–6 in sequence (exchange filter already applied by caller).

    Drops per stage:
        fetch_failure  — yfinance returned None or raised
        market_cap     — below min_market_cap
        liquidity      — avg daily dollar vol (30d) below min_dollar_volume
        fundamentals   — trailing P/E, book value, or revenue missing
        non_equity     — quoteType != 'EQUITY'
        neg_eps        — trailing EPS <= 0 (when include_unprofitable=False)

    Returns (passing_tickers, drop_counts_by_stage).
    """
    passing: list[str] = []
    drops: dict[str, int] = {
        "fetch_failure": 0,
        "market_cap": 0,
        "liquidity": 0,
        "fundamentals": 0,
        "non_equity": 0,
        "neg_eps": 0,
    }

    total = len(tickers)
    w = len(str(total))

    for i, ticker in enumerate(tickers, 1):
        print(f"\r  [{i:{w}}/{total}]  {ticker:<14}", end="", flush=True)

        m = fetch_metrics(
            ticker,
            use_cache=use_cache,
            cache_dir=cache_dir,
            sleep_seconds=sleep_seconds,
            logger=logger,
        )
        if m is None:
            drops["fetch_failure"] += 1
            continue

        # Filter 2: Market cap floor
        mc = m.get("market_cap")
        if mc is None or mc < min_market_cap:
            drops["market_cap"] += 1
            continue

        # Filter 3: Liquidity floor — avg daily dollar vol, last 30 trading days
        dv = m.get("dollar_vol_30d")
        if dv is None or dv < min_dollar_volume:
            drops["liquidity"] += 1
            continue

        # Filter 4: Fundamentals must be present
        pe = m.get("trailing_pe")
        bv = m.get("book_value")
        rev = m.get("total_revenue")
        if pe is None or bv is None or rev is None:
            drops["fundamentals"] += 1
            missing = [
                name
                for name, val in [
                    ("trailing_pe", pe),
                    ("book_value", bv),
                    ("total_revenue", rev),
                ]
                if val is None
            ]
            logger.warning(f"{ticker}: dropped — missing fundamentals: {', '.join(missing)}")
            continue

        # Filter 5: Equity only
        if m.get("quote_type") != "EQUITY":
            drops["non_equity"] += 1
            continue

        # Filter 6: Positive earnings
        if not include_unprofitable:
            eps = m.get("trailing_eps")
            if eps is None or eps <= 0:
                drops["neg_eps"] += 1
                continue

        passing.append(ticker)

    print()  # newline after inline progress
    return passing, drops


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Canadian ticker universe to a value-screening sub-list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input ticker file (one ticker per line)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output ticker file")
    parser.add_argument("--include-venture", action="store_true",
                        help="Include TSX Venture (.V) tickers (excluded by default)")
    parser.add_argument("--min-market-cap", type=float, default=500_000_000, metavar="CAD",
                        help="Minimum market cap in CAD")
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000, metavar="CAD",
                        help="Minimum avg daily dollar volume in CAD over last 30 trading days")
    parser.add_argument("--include-unprofitable", action="store_true",
                        help="Include tickers with negative trailing EPS (excluded by default)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and fetch fresh data")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="Cache directory")
    parser.add_argument("--sleep", type=float, default=0.2, metavar="SEC",
                        help="Seconds to sleep between network fetches")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N tickers (for testing)")
    args = parser.parse_args()

    logger = _setup_logging(Path(DEFAULT_LOG_DIR))
    cache_dir = Path(args.cache_dir)

    # ── Load & exchange filter (free check, no API) ───────────────────────────
    all_tickers = load_tickers(Path(args.input))
    if args.limit is not None:
        all_tickers = all_tickers[: args.limit]

    raw_count = len(all_tickers)

    tickers = filter_exchange(all_tickers, args.include_venture)
    dropped_exchange = raw_count - len(tickers)

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "─" * 54
    print()
    print("Value Universe Builder")
    print(sep)
    print(f"  Input:            {args.input}")
    print(f"  Output:           {args.output}")
    print(f"  Mkt cap floor:    CAD {args.min_market_cap:>14,.0f}")
    print(f"  Dollar vol floor: CAD {args.min_dollar_volume:>14,.0f} / day")
    if args.limit is not None:
        print(f"  [TEST MODE]       first {args.limit} tickers only")
    print(sep)
    print(f"  Input count:      {raw_count}")
    if not args.include_venture:
        print(f"  After exchange:   {len(tickers)}  (dropped {dropped_exchange} .V tickers)")
    print(sep)
    print()

    # ── Filter pipeline ───────────────────────────────────────────────────────
    passing, drops = apply_filters(
        tickers,
        min_market_cap=args.min_market_cap,
        min_dollar_volume=args.min_dollar_volume,
        include_unprofitable=args.include_unprofitable,
        use_cache=not args.no_cache,
        cache_dir=cache_dir,
        sleep_seconds=args.sleep,
        logger=logger,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    save_tickers(passing, Path(args.output))

    # ── Summary table ─────────────────────────────────────────────────────────
    # Build rows: (label, surviving_count, dropped_this_stage)
    rows: list[tuple[str, int, Optional[int]]] = []
    remaining = len(tickers)
    rows.append(("Loaded (post-exchange filter)", remaining, None))

    def _row(label: str, key: str) -> None:
        nonlocal remaining
        d = drops[key]
        remaining -= d
        rows.append((label, remaining, d if d else None))

    _row("After fetch failures", "fetch_failure")
    _row("After market cap filter", "market_cap")
    _row("After liquidity filter", "liquidity")
    _row("After fundamentals check", "fundamentals")
    _row("After equity-only filter", "non_equity")
    if not args.include_unprofitable:
        _row("After profitability filter", "neg_eps")

    col = max(len(r[0]) for r in rows) + 2
    print()
    print(sep)
    print("Filter Summary")
    print(sep)
    for label, count, dropped in rows:
        suffix = f"  (-{dropped})" if dropped else ""
        print(f"  {label:<{col}} {count:>5}{suffix}")
    print(sep)
    print(f"  {'Output':<{col}} {len(passing):>5}")
    print(f"  Written to: {args.output}")
    print()


if __name__ == "__main__":
    main()
