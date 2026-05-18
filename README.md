# Financing (Python utilities)

A small "entry-point" repo that hosts my Python finance utilities and shared helpers used across my other projects.

> **Disclaimer**: This is not financial advice. It's a screening/ranking tool, not a trading system.

---

## Repo layout

```
.
├─ data/
│  ├─ can_tickers                   # curated hand-maintained ticker list
│  ├─ can_tickers_full              # full Yahoo CA market dump (fetch_tickers.py)
│  ├─ can_tickers_swing_universe    # output of swing_universe.py
│  └─ can_tickers_value_universe    # output of value_universe.py
├─ out/
│  ├─ can_tickers_swing_one_line    # comma-separated swing universe
│  └─ can_tickers_rejected.csv      # rejected tickers + reasons
├─ py/
│  ├─ swing_universe.py             # swing trading universe builder
│  ├─ value_universe.py             # value screening universe builder
│  ├─ tickers_info.py               # ticker metadata builder
│  ├─ wheel_screener.py             # Wheel strategy options screener
│  ├─ fetch_tickers.py              # downloads full CA market ticker list
│  ├─ test_swing_universe.py        # 115 tests for swing_universe.py
│  └─ test_value_universe.py        # 40 tests for value_universe.py
├─ requirements.txt
└─ README.md
```

---

## Requirements

- Python 3.10+
- Packages: `yfinance`, `pandas`, `numpy`

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
pip install tabulate         # optional: prettier output in wheel_screener
```

---

## Running the tests

Both test suites run fully offline — yfinance is mocked, no network calls.

```bash
# Swing universe builder — 115 tests
python -m pytest py/test_swing_universe.py -v

# Value universe builder — 40 tests
python -m pytest py/test_value_universe.py -v

# Both suites at once
python -m pytest py/test_swing_universe.py py/test_value_universe.py -v
```

What the swing suite covers: every public function, all 6 filter branches and their
boundary conditions (strict `<`/`>` semantics), all scoring components and caps, gap-risk
ATR capture, relative-strength computation, stale-data tz handling, MultiIndex and
flat-DataFrame yfinance paths, batch-exception recovery, and all three output files.

---

## Swing Universe Builder (`py/swing_universe.py`)

### What it does

Given a text file of tickers (one per line), downloads 1 year of daily OHLCV history and
applies a filter + scoring pipeline to identify Canadian stocks suitable for **1–4 week
swing trades**. Outputs a ranked universe file suitable for feeding into other scanners.

### Pipeline

1. Loads tickers from input file (deduplicates, strips comments)
2. Fetches benchmark (XIU.TO) once for relative-strength comparison
3. Downloads history in batches of 80 (sleeps between batches to avoid rate limits)
4. Per ticker: ATR-14, dollar volume, SMA50/200, worst 1-day return, SMA50 slope,
   RS vs benchmark, volume trend
5. Hard filters (`Thresholds`): price, dollar volume, ATR%, worst-day drop, above SMA50,
   stale data — tickers failing any are written to the rejected CSV with reasons
6. Scores passing tickers across 7 dimensions and sorts descending

### Filters (hard gates)

| Threshold | Default | Rationale |
|-----------|---------|-----------|
| `min_price` | $1.00 | Excludes penny stocks and manipulation-prone names |
| `min_avg_dollar_vol_20` | $1,000,000 | Enough liquidity to enter/exit without slippage |
| `max_atr_pct_14` | 5% | ATR > 5% means stops are impractically wide for 1–4w swings |
| `max_one_day_drop_126` | −15% | Screens out gap-down risk; one day ≥ 15% drop is a red flag |
| `require_above_50d` | True | Only trade in direction of intermediate trend |
| `max_stale_days` | 5 | Reject tickers with no recent price data |

### Scoring components

| Component | Max pts | Notes |
|-----------|---------|-------|
| Liquidity (log scale) | +3.0 | log10(dollar_vol) − 5, capped |
| Above SMA50 | +1.0 | Hard filter already gates this |
| Above SMA200 | +2.0 | Strong trend bonus (FIX #2: raised from +0.8) |
| SMA50 slope | +1.5 | Normalized by price, capped |
| RS 1-month vs XIU | +2.0 / −2.0 | Outperformance vs benchmark |
| RS 3-month vs XIU | +1.5 / −1.5 | Confirmation of RS signal |
| Volume trend | +1.3 | +0.8 base + up to +0.5 for strong accumulation |
| ATR penalty | −2.0 | 0.05 × 15 = 0.75 max for a passing stock |
| Worst-day penalty | −2.5 | 0.15 × 10 = 1.5 max for a passing stock |

### How to run

```bash
# Default: reads data/can_tickers_full, writes data/can_tickers_swing_universe
cd py && python swing_universe.py

# Custom paths
cd py && python swing_universe.py \
    --input ../data/can_tickers \
    --out ../data/can_tickers_swing_universe \
    --out-one-line ../out/can_tickers_swing_one_line \
    --out-rejected ../out/can_tickers_rejected.csv
```

### From code (IDE / other scripts)

```python
from swing_universe import UniverseBuilderConfig, Thresholds, run_universe_builder

cfg = UniverseBuilderConfig(
    tickers_path="../data/can_tickers",
    benchmark="XIU.TO",
    out_file_path="../data/can_tickers_swing_universe",
    out_one_line_file_path="../out/can_tickers_swing_one_line",
    out_rejected_file_path="../out/can_tickers_rejected.csv",
    period="1y",
    interval="1d",
    auto_adjust=True,
    batch_size=80,
    sleep_seconds=1.0,
    thresholds=Thresholds(
        min_price=1.0,
        min_avg_dollar_vol_20=1_000_000.0,
        max_atr_pct_14=0.05,
        max_one_day_drop_126=-0.15,
        require_above_50d=True,
        prefer_above_200d=True,
        max_stale_days=5,
    ),
)

df_tradable, df_rejected = run_universe_builder(cfg)
print(df_tradable.head())
```

### Outputs

| File | Format | Contents |
|------|--------|----------|
| `data/can_tickers_swing_universe` | one ticker per line | Ranked tradable universe — load directly into other tools |
| `out/can_tickers_swing_one_line` | single comma-separated line | Quick copy/paste format |
| `out/can_tickers_rejected.csv` | CSV | Rejected tickers + `reject_reasons` column for diagnostics |

Tip: when tuning thresholds, the rejected CSV is the fastest way to see *why* names are failing.
Filter rejection reasons: `price_too_low`, `low_dollar_volume`, `too_volatile_atr`,
`large_gap_risk`, `below_50d`, `stale_data_Nd`.

---

## Value Universe Builder (`py/value_universe.py`)

### What it does

Filters the full Canadian ticker universe (`data/can_tickers_full`, ~4,700 tickers) down to
a sub-list of quality, liquid, profitable equities suitable for downstream value factor
screening. Not a value scorer — just a clean input list.

### How to run

```bash
# Default run (from repo root)
python py/value_universe.py

# Test on first 50 tickers
python py/value_universe.py --limit 50

# Force fresh data, skip cache
python py/value_universe.py --no-cache

# Custom thresholds
python py/value_universe.py --min-market-cap 1000000000 --min-dollar-volume 5000000
```

Output is written to `data/can_tickers_value_universe` (one ticker per line).
Errors are logged to `logs/value_universe_errors.log`.
yfinance responses are cached in `.cache/yfinance/` with a 24-hour TTL.

### Filters (applied in order)

| # | Filter | Default | Why |
|---|--------|---------|-----|
| 1 | **Exchange** — drop `.V` (TSX Venture) | enabled | Micro-caps with thin liquidity and sparse data |
| 2 | **Market cap floor** | CAD 500M | Analyst coverage and fundamental data quality |
| 3 | **Liquidity floor** | CAD 1M avg daily dollar vol (30d) | Minimum tradability |
| 4 | **Fundamentals present** | trailing P/E, book value, revenue | Value ratios require these; missing = data quality issue |
| 5 | **Equity only** | `quoteType == EQUITY` | Drops ETFs, closed-end funds, trusts |
| 6 | **Positive earnings** | trailing EPS > 0 | Negative EPS makes P/E undefined |

Use `--include-venture`, `--include-unprofitable`, and the threshold flags to relax filters.

---

## Ticker Metadata Builder (`py/tickers_info.py`)

Fetches Yahoo Finance metadata for a ticker list and writes a CSV with exchange, company
name, sector, volume, market cap, price, and spread estimate.

```bash
python py/tickers_info.py --input data/can_tickers --out out/can_tickers_info.csv
```

Optional: `--fallback-spread-pct` (default `0.01`), `--batch-size` (default `80`),
`--sleep` (default `1.0` s).

---

## Wheel Strategy Screener (`py/wheel_screener.py`)

Screens CA + US stocks for the Wheel strategy (sell Cash-Secured Puts → Covered Calls).
Scores across 6 pillars (profitability, FCF, balance sheet, valuation, growth, options chain)
and assigns tiers: STRONG / SOLID / WATCH / SKIP.

```bash
cd py && python wheel_screener.py
```

Outputs `wheel_candidates.csv`, `wheel_all_screened.csv`, `wheel_candidates.json`
(relative to `py/`).

---

## Canadian Market Download (`py/fetch_tickers.py`)

Downloads the full Yahoo Finance Canadian equity universe and writes it to
`data/can_tickers_full` (used as input for `swing_universe.py` and `value_universe.py`).

```bash
cd py && python fetch_tickers.py
```

---

## Notes on data quality

- `auto_adjust=True` is used in all `yf.download()` calls to eliminate split/dividend
  artifacts from SMA, ATR, and return calculations.
- `yf.download()` with multiple tickers returns a `pd.MultiIndex` DataFrame (level 0 =
  ticker); single-ticker responses return a flat DataFrame — both paths are handled.
- Yahoo data can be missing or inconsistent; the code guards against common failure modes
  and tracks every rejection with a typed reason.

---

## Related projects

- Stage Radar: [https://github.com/ChernyshovYuriy/stage-radar](https://github.com/ChernyshovYuriy/stage-radar)
- Point & Figure System: [https://github.com/ChernyshovYuriy/pfsystem](https://github.com/ChernyshovYuriy/pfsystem)
- TSX Canadian Stock Screener: [https://github.com/ChernyshovYuriy/stock-scanner](https://github.com/ChernyshovYuriy/stock-scanner)

---

## License

MIT
