"""
Tests for value_universe.py.

Covers every filter branch and all I/O helpers with no real network calls.
Run from repo root:
    pytest py/test_value_universe.py -v
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from value_universe import (
    CACHE_TTL_HOURS,
    _cache_path,
    _load_cache,
    _save_cache,
    apply_filters,
    fetch_metrics,
    filter_exchange,
    load_tickers,
    save_tickers,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_value_universe")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def _metrics(**overrides) -> dict:
    """Return a fully valid metrics dict that passes all filters by default."""
    base: dict = {
        "ticker": "RY.TO",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "market_cap": 200_000_000_000,   # $200B CAD
        "quote_type": "EQUITY",
        "trailing_pe": 12.0,
        "book_value": 50.0,
        "total_revenue": 20_000_000_000,
        "trailing_eps": 8.0,
        "dollar_vol_30d": 50_000_000,    # $50M/day
    }
    base.update(overrides)
    return base


# ─── load_tickers ─────────────────────────────────────────────────────────────


def test_load_tickers_basic(tmp_path: Path) -> None:
    f = tmp_path / "tickers.txt"
    f.write_text("RY.TO\nTD.TO\nBNS.TO\n")
    assert load_tickers(f) == ["RY.TO", "TD.TO", "BNS.TO"]


def test_load_tickers_deduplicates(tmp_path: Path) -> None:
    f = tmp_path / "tickers.txt"
    f.write_text("RY.TO\nTD.TO\nRY.TO\nTD.TO\n")
    assert load_tickers(f) == ["RY.TO", "TD.TO"]


def test_load_tickers_strips_blanks_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "tickers.txt"
    f.write_text("  RY.TO  \n\n# comment\nTD.TO\n")
    assert load_tickers(f) == ["RY.TO", "TD.TO"]


def test_load_tickers_uppercases(tmp_path: Path) -> None:
    f = tmp_path / "tickers.txt"
    f.write_text("ry.to\ntd.to\n")
    assert load_tickers(f) == ["RY.TO", "TD.TO"]


# ─── save_tickers ─────────────────────────────────────────────────────────────


def test_save_tickers_one_per_line(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    save_tickers(["RY.TO", "TD.TO", "BNS.TO"], out)
    assert out.read_text() == "RY.TO\nTD.TO\nBNS.TO\n"


def test_save_tickers_empty_list(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    save_tickers([], out)
    assert out.read_text() == ""


def test_save_tickers_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "out.txt"
    save_tickers(["RY.TO"], out)
    assert out.exists()


# ─── filter_exchange ──────────────────────────────────────────────────────────


def test_filter_exchange_drops_venture() -> None:
    tickers = ["RY.TO", "WEED.V", "SHOP.TO", "MICRO.V", "AAPL.NE"]
    assert filter_exchange(tickers, include_venture=False) == ["RY.TO", "SHOP.TO", "AAPL.NE"]


def test_filter_exchange_keeps_venture_when_flag_set() -> None:
    tickers = ["RY.TO", "WEED.V", "SHOP.TO"]
    assert filter_exchange(tickers, include_venture=True) == tickers


def test_filter_exchange_preserves_order() -> None:
    tickers = ["Z.TO", "A.TO", "M.TO"]
    assert filter_exchange(tickers, include_venture=False) == ["Z.TO", "A.TO", "M.TO"]


def test_filter_exchange_does_not_drop_cn_or_ne() -> None:
    tickers = ["NVDA.NE", "AAA.CN", "SHOP.TO"]
    assert filter_exchange(tickers, include_venture=False) == tickers


# ─── cache ────────────────────────────────────────────────────────────────────


def test_cache_roundtrip(cache_dir: Path) -> None:
    data = _metrics(ticker="BNS.TO")
    _save_cache("BNS.TO", data, cache_dir)
    result = _load_cache("BNS.TO", cache_dir)
    assert result is not None
    assert result["ticker"] == "BNS.TO"
    assert result["market_cap"] == data["market_cap"]


def test_cache_hit_within_ttl(cache_dir: Path) -> None:
    data = _metrics()
    _save_cache("RY.TO", data, cache_dir)
    assert _load_cache("RY.TO", cache_dir) is not None


def test_cache_miss_after_ttl_expiry(cache_dir: Path) -> None:
    expired_time = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS + 1)
    data = _metrics(fetched_at=expired_time.isoformat())
    _save_cache("RY.TO", data, cache_dir)
    assert _load_cache("RY.TO", cache_dir) is None


def test_cache_miss_when_file_absent(cache_dir: Path) -> None:
    assert _load_cache("NOTEXIST.TO", cache_dir) is None


def test_cache_returns_none_on_corrupt_json(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path("BAD.TO", cache_dir).write_text("not valid json", encoding="utf-8")
    assert _load_cache("BAD.TO", cache_dir) is None


def test_cache_handles_nan_values(cache_dir: Path) -> None:
    data = _metrics(dollar_vol_30d=float("nan"))
    _save_cache("RY.TO", data, cache_dir)
    result = _load_cache("RY.TO", cache_dir)
    assert result is not None
    assert result["dollar_vol_30d"] is None  # NaN serialised as null


def test_cache_filename_escapes_dots(cache_dir: Path) -> None:
    p = _cache_path("SHOP.TO", cache_dir)
    assert ".TO" not in p.name
    assert p.name == "SHOP_TO.json"


# ─── fetch_metrics ────────────────────────────────────────────────────────────


def _mock_ticker(info: dict, history_rows: int = 30) -> MagicMock:
    """Build a mock yf.Ticker with controllable .info and .history()."""
    mock_t = MagicMock()
    mock_t.info = info
    hist = pd.DataFrame({
        "Close": [20.0] * history_rows,
        "Volume": [500_000] * history_rows,
    })
    mock_t.history.return_value = hist
    return mock_t


@patch("value_universe.yf.Ticker")
def test_fetch_metrics_maps_fields_correctly(mock_cls: MagicMock, cache_dir: Path, logger: logging.Logger) -> None:
    mock_cls.return_value = _mock_ticker({
        "marketCap": 100_000_000_000,
        "quoteType": "EQUITY",
        "trailingPE": 15.0,
        "bookValue": 40.0,
        "totalRevenue": 5_000_000_000,
        "trailingEps": 4.5,
    })
    result = fetch_metrics("RY.TO", use_cache=False, cache_dir=cache_dir, sleep_seconds=0, logger=logger)

    assert result is not None
    assert result["market_cap"] == 100_000_000_000
    assert result["quote_type"] == "EQUITY"
    assert result["trailing_pe"] == 15.0
    assert result["book_value"] == 40.0
    assert result["total_revenue"] == 5_000_000_000
    assert result["trailing_eps"] == 4.5
    assert result["dollar_vol_30d"] == pytest.approx(20.0 * 500_000)


@patch("value_universe.yf.Ticker")
def test_fetch_metrics_returns_none_on_exception(mock_cls: MagicMock, cache_dir: Path, logger: logging.Logger) -> None:
    mock_cls.side_effect = Exception("network error")
    assert fetch_metrics("BAD.TO", use_cache=False, cache_dir=cache_dir, sleep_seconds=0, logger=logger) is None


@patch("value_universe.yf.Ticker")
def test_fetch_metrics_writes_cache(mock_cls: MagicMock, cache_dir: Path, logger: logging.Logger) -> None:
    mock_cls.return_value = _mock_ticker({"marketCap": 1_000_000_000, "quoteType": "EQUITY",
                                          "trailingPE": 10.0, "bookValue": 5.0,
                                          "totalRevenue": 500_000_000, "trailingEps": 1.0})
    fetch_metrics("TD.TO", use_cache=True, cache_dir=cache_dir, sleep_seconds=0, logger=logger)
    assert _cache_path("TD.TO", cache_dir).exists()


@patch("value_universe.yf.Ticker")
def test_fetch_metrics_uses_cache_hit(mock_cls: MagicMock, cache_dir: Path, logger: logging.Logger) -> None:
    cached = _metrics(ticker="RY.TO")
    _save_cache("RY.TO", cached, cache_dir)

    fetch_metrics("RY.TO", use_cache=True, cache_dir=cache_dir, sleep_seconds=0, logger=logger)
    mock_cls.assert_not_called()


@patch("value_universe.yf.Ticker")
def test_fetch_metrics_history_failure_yields_none_dollar_vol(mock_cls: MagicMock, cache_dir: Path, logger: logging.Logger) -> None:
    mock_t = MagicMock()
    mock_t.info = {"marketCap": 1_000_000_000, "quoteType": "EQUITY",
                   "trailingPE": 10.0, "bookValue": 5.0,
                   "totalRevenue": 500_000_000, "trailingEps": 1.0}
    mock_t.history.side_effect = Exception("history unavailable")
    mock_cls.return_value = mock_t

    result = fetch_metrics("RY.TO", use_cache=False, cache_dir=cache_dir, sleep_seconds=0, logger=logger)
    assert result is not None
    assert result["dollar_vol_30d"] is None


# ─── apply_filters ────────────────────────────────────────────────────────────


def _run_filters(tickers: list[str], fetch_side_effect, **overrides) -> tuple[list[str], dict[str, int]]:
    """Patch fetch_metrics and run apply_filters with sensible defaults."""
    defaults = dict(
        min_market_cap=500_000_000,
        min_dollar_volume=1_000_000,
        include_unprofitable=False,
        use_cache=False,
        cache_dir=Path("/tmp/unused"),
        sleep_seconds=0,
        logger=logging.getLogger("test"),
    )
    defaults.update(overrides)
    with patch("value_universe.fetch_metrics", side_effect=fetch_side_effect):
        return apply_filters(tickers, **defaults)


def test_apply_filters_passes_all_valid_tickers() -> None:
    tickers = ["RY.TO", "TD.TO", "BNS.TO"]
    passing, drops = _run_filters(tickers, lambda t, **kw: _metrics(ticker=t))
    assert passing == tickers
    assert all(v == 0 for v in drops.values())


def test_apply_filters_fetch_failure_drops_ticker() -> None:
    passing, drops = _run_filters(["BAD.TO"], lambda t, **kw: None)
    assert passing == []
    assert drops["fetch_failure"] == 1


def test_apply_filters_market_cap_none_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(market_cap=None))
    assert passing == []
    assert drops["market_cap"] == 1


def test_apply_filters_market_cap_below_threshold_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(market_cap=499_999_999))
    assert passing == []
    assert drops["market_cap"] == 1


def test_apply_filters_market_cap_at_threshold_passes() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(market_cap=500_000_000))
    assert passing == ["X.TO"]
    assert drops["market_cap"] == 0


def test_apply_filters_dollar_vol_none_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(dollar_vol_30d=None))
    assert passing == []
    assert drops["liquidity"] == 1


def test_apply_filters_dollar_vol_below_threshold_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(dollar_vol_30d=999_999))
    assert passing == []
    assert drops["liquidity"] == 1


def test_apply_filters_missing_pe_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(trailing_pe=None))
    assert passing == []
    assert drops["fundamentals"] == 1


def test_apply_filters_missing_book_value_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(book_value=None))
    assert passing == []
    assert drops["fundamentals"] == 1


def test_apply_filters_missing_revenue_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(total_revenue=None))
    assert passing == []
    assert drops["fundamentals"] == 1


def test_apply_filters_non_equity_drops_ticker() -> None:
    for qt in ("ETF", "MUTUALFUND", "CURRENCY", None):
        passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(quote_type=qt))
        assert passing == [], f"expected drop for quoteType={qt!r}"
        assert drops["non_equity"] == 1


def test_apply_filters_negative_eps_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(trailing_eps=-0.01))
    assert passing == []
    assert drops["neg_eps"] == 1


def test_apply_filters_zero_eps_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(trailing_eps=0.0))
    assert passing == []
    assert drops["neg_eps"] == 1


def test_apply_filters_none_eps_drops_ticker() -> None:
    passing, drops = _run_filters(["X.TO"], lambda t, **kw: _metrics(trailing_eps=None))
    assert passing == []
    assert drops["neg_eps"] == 1


def test_apply_filters_include_unprofitable_keeps_negative_eps() -> None:
    passing, drops = _run_filters(
        ["X.TO"],
        lambda t, **kw: _metrics(trailing_eps=-5.0),
        include_unprofitable=True,
    )
    assert passing == ["X.TO"]
    assert drops["neg_eps"] == 0


def test_apply_filters_each_ticker_drops_at_exactly_one_stage() -> None:
    """Drop counts must sum to len(input) - len(passing)."""
    def fetch(t: str, **kw) -> dict | None:
        return {
            "FAIL.TO":    None,
            "SMALL.TO":   _metrics(market_cap=1),
            "ILLIQ.TO":   _metrics(dollar_vol_30d=1),
            "NOFUND.TO":  _metrics(trailing_pe=None),
            "ETF.TO":     _metrics(quote_type="ETF"),
            "LOSS.TO":    _metrics(trailing_eps=-1.0),
            "GOOD.TO":    _metrics(ticker="GOOD.TO"),
        }[t]

    tickers = ["FAIL.TO", "SMALL.TO", "ILLIQ.TO", "NOFUND.TO", "ETF.TO", "LOSS.TO", "GOOD.TO"]
    passing, drops = _run_filters(tickers, fetch)

    assert passing == ["GOOD.TO"]
    total_dropped = sum(drops.values())
    assert total_dropped == len(tickers) - len(passing)
    assert drops["fetch_failure"] == 1
    assert drops["market_cap"] == 1
    assert drops["liquidity"] == 1
    assert drops["fundamentals"] == 1
    assert drops["non_equity"] == 1
    assert drops["neg_eps"] == 1


def test_apply_filters_drop_order_market_cap_before_liquidity() -> None:
    """A ticker failing market cap must not be counted in liquidity drops."""
    passing, drops = _run_filters(
        ["X.TO"],
        lambda t, **kw: _metrics(market_cap=1, dollar_vol_30d=1),
    )
    assert drops["market_cap"] == 1
    assert drops["liquidity"] == 0
