"""
╔══════════════════════════════════════════════════════════════════════╗
║          🎡  WHEEL STRATEGY STOCK SCREENER                          ║
║          Canada (TSX) + USA  |  Fundamental Quality Filter          ║
╚══════════════════════════════════════════════════════════════════════╝

Screens stocks suitable for the Options Wheel Strategy (CSP → CC cycle)
using strong, defensible fundamental criteria.

HOW IT WORKS:
  1. Pulls live fundamental data via yfinance for 130+ CA + US stocks
  2. Scores each stock across 5 core pillars (0–100%)
  3. Assigns a Wheel tier: 🟢 STRONG / 🟡 SOLID / 🟠 WATCH / 🔴 SKIP
  4. Outputs a ranked table + detailed scorecards + CSV/JSON files

INSTALL DEPENDENCIES (run once):
  pip install yfinance pandas tabulate

RUN:
  python wheel_screener.py

CUSTOMIZE:
  Adjust THRESHOLDS dict or run_screener() parameters at the bottom.

DISCLAIMER:
  This is a research/educational tool. Not financial advice.
  Always verify options liquidity before trading.

Author: Claude (Anthropic)  |  Uses: yfinance, pandas, tabulate
"""

import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("wheel_screener")

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("TIP: pip install tabulate  for prettier tables\n")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE
# Quality-biased candidates well-suited for the Wheel Strategy.
# Add or remove tickers as you see fit.
# Canadian tickers use the .TO suffix (TSX); .V for TSX Venture.
# ══════════════════════════════════════════════════════════════════════

CANADIAN_STOCKS_BASE = {
    "CA_Banks": ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"],
    "CA_Insurance": ["MFC.TO", "SLF.TO", "IFC.TO", "FFH.TO"],
    "CA_Energy": ["CNQ.TO", "SU.TO", "CVE.TO", "ARX.TO", "TOU.TO"],
    "CA_Pipelines": ["ENB.TO", "TRP.TO", "PPL.TO", "KEY.TO"],
    "CA_Industrials": ["CNR.TO", "CP.TO", "WSP.TO", "STN.TO", "TIH.TO"],
    "CA_Consumer": ["ATD.TO", "DOL.TO", "L.TO", "MRU.TO", "WN.TO"],
    "CA_Telecom": ["BCE.TO", "T.TO", "RCI-B.TO"],
    "CA_Tech": ["CSU.TO", "SHOP.TO", "KXS.TO", "DSG.TO", "OTEX.TO"],
    "CA_REITS": ["BEI-UN.TO", "CAR-UN.TO", "DIR-UN.TO", "GRT-UN.TO"],
    "CA_Diversified": ["BAM.TO", "BN.TO", "POW.TO", "GWO.TO"],
    "CA_Materials": ["NTR.TO", "AGI.TO", "WPM.TO", "FNV.TO"],
    "CA_Utilities": ["FTS.TO", "AQN.TO", "EMA.TO", "H.TO"],
}

US_STOCKS = {
    "US_Tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM", "ADBE", "INTC", "TXN"],
    "US_Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BRK-B"],
    "US_Healthcare": ["JNJ", "UNH", "ABT", "TMO", "DHR", "ABBV", "MRK", "LLY", "BMY", "MDT"],
    "US_Consumer_Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MCD", "YUM", "MDLZ"],
    "US_Consumer_Disc": ["HD", "LOW", "TJX", "AMZN", "NKE", "SBUX", "TGT"],
    "US_Industrials": ["UPS", "FDX", "CAT", "DE", "MMM", "HON", "GE", "RTX", "LMT", "NOC"],
    "US_Energy": ["XOM", "CVX", "COP", "EOG", "PSX", "MPC", "SLB", "OXY"],
    "US_Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "US_REITS": ["PLD", "AMT", "EQIX", "PSA", "WPC", "O", "VICI", "DLR"],
    "US_Materials": ["LIN", "APD", "NEM", "FCX", "NUE", "VMC", "MLM"],
}

CANADIAN_STOCKS = {

    # 🏦 Banks (core liquidity / options heavy)
    "CA_Banks": [
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"
    ],

    # 🛡️ Insurance / Financial Services
    "CA_Insurance": [
        "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "POW.TO", "IGM.TO"
    ],

    # 🛢️ Energy – Integrated / Pipelines
    "CA_Energy_Majors": [
        "SU.TO", "CNQ.TO", "CVE.TO", "IMO.TO",
        "ENB.TO", "TRP.TO", "PPL.TO", "KEY.TO"
    ],

    # 🛢️ Energy – E&P (higher beta, wheel candidates)
    "CA_Energy_E&P": [
        "ARX.TO", "TOU.TO", "WCP.TO", "TVE.TO", "PEY.TO",
        "BTE.TO", "AAV.TO", "VET.TO", "CJ.TO", "SGY.TO",
        "OBE.TO", "PHX.TO", "POU.TO"
    ],

    # ⛏️ Gold / Precious Metals
    "CA_Gold": [
        "ABX.TO", "AEM.TO", "K.TO", "FNV.TO", "WPM.TO",
        "EQX.TO", "SSRM.TO", "AGI.TO", "OGC.TO"
    ],

    # ⛏️ Silver / Precious
    "CA_Silver": [
        "PAAS.TO", "AG.TO", "SVM.TO"
    ],

    # ⛏️ Base Metals / Diversified Mining
    "CA_BaseMetals": [
        "TECK-B.TO", "FM.TO", "LUN.TO", "HBM.TO",
        "ERO.TO", "CS.TO", "NG.TO", "LUG.TO"
    ],

    # ⛏️ Uranium / Nuclear
    "CA_Uranium": [
        "CCO.TO", "EFR.TO", "NXE.TO"
    ],

    # ⛏️ High-growth / Spec Mining
    "CA_Mining_Spec": [
        "IVN.TO", "GMIN.TO", "DPM.TO", "EDR.TO", "FVI.TO"
    ],

    # 🧪 Materials / Fertilizer
    "CA_Materials": [
        "NTR.TO"
    ],

    # 📡 Telecom
    "CA_Telecom": [
        "T.TO", "BCE.TO", "RCI-B.TO", "QBR-B.TO"
    ],

    # ⚡ Utilities / Power
    "CA_Utilities": [
        "FTS.TO", "EMA.TO", "ALA.TO", "NPI.TO", "CPX.TO", "CU.TO"
    ],

    # 🚂 Industrials / Transport
    "CA_Industrials": [
        "CNR.TO", "CP.TO", "WSP.TO", "CAE.TO", "ATRL.TO"
    ],

    # 🛍️ Consumer / Retail
    "CA_Consumer": [
        "ATD.TO", "DOL.TO", "MRU.TO", "CTC-A.TO",
        "EMP-A.TO", "NWC.TO"
    ],

    # 🍔 Restaurants / Food
    "CA_Food": [
        "QSR.TO"
    ],

    # 🧠 Tech
    "CA_Tech": [
        "SHOP.TO", "OTEX.TO", "GIB-A.TO", "CLS.TO"
    ],

    # 🏗️ Infrastructure / Asset Mgmt
    "CA_Asset_Managers": [
        "BAM.TO", "BN.TO", "BIP-UN.TO", "BIPC.TO", "BLX.TO"
    ],

    # 🏢 REITs (income / wheel-friendly)
    "CA_REITs": [
        "REI-UN.TO", "HR-UN.TO", "DIR-UN.TO", "CAR-UN.TO",
        "FCR-UN.TO", "CRT-UN.TO", "PMZ-UN.TO", "CHP-UN.TO"
    ],

    # 🪙 Commodities / Trusts
    "CA_Commodity_Trusts": [
        "PSLV.TO", "PHYS.TO", "U-UN.TO"
    ],

    # 🪙 Crypto / Alt
    "CA_Crypto": [
        "HUT.TO"
    ],

    # ✈️ Aerospace / Defense
    "CA_Aerospace": [
        "CAE.TO", "MDA.TO"
    ],

    # 🛫 Travel / Leisure
    "CA_Travel": [
        "AC.TO"
    ],

    # 🧬 Misc / Small Caps / Mixed
    "CA_Mixed": [
        "AYA.TO", "CEU.TO", "ARIS.TO", "SOBO.TO", "H.TO",
        "TA.TO", "L.TO", "PXT.TO", "GTWO.TO", "HWX.TO",
        "AII.TO", "OLA.TO", "KNT.TO", "GEI.TO", "FRU.TO",
        "SAP.TO", "AAUC.TO", "EFX.TO", "WDO.TO", "TPZ.TO",
        "CG.TO", "EFN.TO", "X.TO", "SGD.TO", "PSK.TO",
        "SES.TO", "EDV.TO", "TRI.TO", "ARE.TO", "TXG.TO",
        "ELD.TO", "SDE.TO", "CDE.TO", "ATZ.TO", "GIL.TO",
        "SSRM.TO", "VNP.TO", "TCW.TO", "CPKR.TO", "FTT.TO",
        "EXE.TO", "LB.TO", "RSI.TO", "PSI.TO", "MER.TO",
        "MDI.TO", "ACO-X.TO", "TIH.TO", "SOIL.TO", "PD.TO",
        "BEP-UN.TO", "KEL.TO", "AGF-B.TO", "BIR.TO",
        "SIA.TO", "MFI.TO", "CRR-UN.TO", "VLE.TO", "JOY.TO"
    ],

    # 🇺🇸 CDRs / Synthetic exposure (important for your system)
    "CA_CDRs": [
        "NVDA.NE", "TSLA.NE", "NFLX.NE"
    ]

}

# ══════════════════════════════════════════════════════════════════════
# SCORING THRESHOLDS  (tweak to match your strategy preferences)
# ══════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    # ── Valuation ────────────────────────────────────────────────────
    "pe_max": 35,  # Max acceptable trailing P/E
    "pe_fair": 20,  # "Cheap" P/E for bonus points
    "pb_max": 8,  # Max Price-to-Book
    "ev_ebitda_max": 20,  # Max EV/EBITDA
    "ev_ebitda_fair": 12,  # "Cheap" EV/EBITDA

    # ── Profitability ─────────────────────────────────────────────────
    "roe_min": 10,  # Minimum Return on Equity (%)
    "roe_good": 18,  # Strong ROE threshold (bonus)
    "profit_margin_min": 5,  # Minimum net profit margin (%)
    "fcf_yield_min": 2,  # Minimum FCF yield (%)

    # ── Balance Sheet ─────────────────────────────────────────────────
    "debt_equity_max": 2.5,  # Max Debt-to-Equity ratio
    "debt_equity_good": 1.0,  # Low-debt bonus threshold
    "current_ratio_min": 1.0,  # Liquidity floor

    # ── Growth ────────────────────────────────────────────────────────
    "revenue_growth_min": 0,  # Must be growing (or flat)
    "earnings_growth_min": 0,  # Must be growing (or flat)

    # ── Dividend (stability proxy) ────────────────────────────────────
    "div_yield_bonus": 1.5,  # Dividend yield % that earns bonus

    # ── Wheel-specific price range ────────────────────────────────────
    "min_price": 10,  # Floor: avoid penny-stock dynamics
    "max_price": 250,  # Ceiling: keep CSP margin manageable
    "min_mktcap_B": 2,  # Min market cap in $B (liquidity)

    # ── Options chain requirements ─────────────────────────────────────
    "min_dte": 20,  # Minimum DTE for candidate expiration
    "max_dte": 50,  # Maximum DTE
    "target_delta": 0.30,  # Target delta for CSP (article uses ~0.28)
    "max_delta": 0.40,  # Max delta (too close to ATM)
    "min_open_interest": 100,  # Minimum OI for liquidity
    "max_bid_ask_pct": 5.0,  # Max bid-ask spread as % of mark
    "min_earnings_dte": 10,  # Min days between expiration and earnings
}


# ══════════════════════════════════════════════════════════════════════
# DATA FETCHER
# ══════════════════════════════════════════════════════════════════════

def _pct_or_none(val):
    """Convert a decimal ratio to percentage, or None if missing."""
    if val is None:
        return None
    return round(val * 100, 2)


def fetch_stock_data(ticker: str) -> dict | None:
    """Fetch and normalise all relevant fundamental data for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        if not info:
            return None

        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            fi = t.fast_info
            price = getattr(fi, "last_price", None)
        if not price or price <= 0:
            return None

        mktcap = info.get("marketCap") or 0

        data = {
            "ticker": ticker,
            "name": (info.get("shortName") or info.get("longName") or ticker)[:30],
            "sector": info.get("sector", "N/A"),
            "industry": (info.get("industry") or "N/A")[:35],
            "currency": info.get("currency", "USD"),
            "price": round(float(price), 2),
            "mktcap_B": round(mktcap / 1e9, 2),

            # ── Valuation ────────────────────────────────────────────
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ps": info.get("priceToSalesTrailing12Months"),

            # ── Profitability ─────────────────────────────────────────
            "roe": (info.get("returnOnEquity") or 0) * 100,
            "roa": (info.get("returnOnAssets") or 0) * 100,
            "profit_margin": (info.get("profitMargins") or 0) * 100,
            "gross_margin": (info.get("grossMargins") or 0) * 100,
            "operating_margin": (info.get("operatingMargins") or 0) * 100,

            # ── Cash Flow ─────────────────────────────────────────────
            "fcf": info.get("freeCashflow"),
            "operating_cf": info.get("operatingCashflow"),
            "total_revenue": info.get("totalRevenue"),

            # ── Debt / Liquidity ──────────────────────────────────────
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "total_debt": info.get("totalDebt"),

            # ── Growth ────────────────────────────────────────────────
            "revenue_growth": _pct_or_none(info.get("revenueGrowth")),
            "earnings_growth": _pct_or_none(info.get("earningsGrowth")),
            "earnings_quarterly_growth": _pct_or_none(info.get("earningsQuarterlyGrowth")),

            # ── Dividend ──────────────────────────────────────────────
            "div_yield": (info.get("dividendYield") or 0) * 100,
            "payout_ratio": (info.get("payoutRatio") or 0) * 100,

            # ── Analyst signals ───────────────────────────────────────
            "target_price": info.get("targetMeanPrice"),
            "analyst_rating": info.get("recommendationMean"),  # 1=Strong Buy → 5=Sell
            "num_analysts": info.get("numberOfAnalystOpinions", 0),

            # ── Market / Options context ──────────────────────────────
            "beta": info.get("beta"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low": info.get("fiftyTwoWeekLow"),
        }

        # FCF yield
        if data["fcf"] and mktcap > 0:
            data["fcf_yield"] = round((data["fcf"] / mktcap) * 100, 2)
        else:
            data["fcf_yield"] = None

        # yfinance returns debtToEquity as percentage (e.g. 150 = 1.5x)
        de = data["debt_equity"]
        if de is not None:
            data["debt_equity"] = de / 100

        # Analyst upside
        if data["target_price"] and price > 0:
            data["upside_pct"] = round(((data["target_price"] - price) / price) * 100, 1)
        else:
            data["upside_pct"] = None

        return data

    except Exception as e:
        logger.warning(f"{ticker}: failed to fetch data — {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# OPTIONS CHAIN VALIDATION
# ══════════════════════════════════════════════════════════════════════

def fetch_options_data(ticker: str, price: float) -> dict | None:
    """
    Find the best CSP candidate from the options chain.
    Returns dict with IV, delta, OI, bid-ask spread, ROC, DTE, etc.
    Returns None if no suitable expiration/strike exists.
    """
    T = THRESHOLDS
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None

        today = datetime.now().date()
        best = None

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte < T["min_dte"] or dte > T["max_dte"]:
                continue

            chain = t.option_chain(exp_str)
            puts = chain.puts
            if puts.empty:
                continue

            # Find strikes in the OTM zone (below current price)
            otm_puts = puts[puts["strike"] < price].copy()
            if otm_puts.empty:
                continue

            # Calculate distance from price (as pct OTM)
            otm_puts = otm_puts.copy()
            otm_puts["pct_otm"] = ((price - otm_puts["strike"]) / price) * 100

            # Target ~5-10% OTM as the article suggests
            candidates = otm_puts[
                (otm_puts["pct_otm"] >= 3) & (otm_puts["pct_otm"] <= 15)
                ]
            if candidates.empty:
                candidates = otm_puts

            for _, opt in candidates.iterrows():
                strike = opt["strike"]
                bid = opt.get("bid", 0) or 0
                ask = opt.get("ask", 0) or 0
                oi = opt.get("openInterest", 0) or 0
                iv = opt.get("impliedVolatility", 0) or 0

                if bid <= 0 or ask <= 0:
                    continue
                if oi < T["min_open_interest"]:
                    continue

                mark = (bid + ask) / 2
                spread_pct = ((ask - bid) / mark) * 100 if mark > 0 else 999

                if spread_pct > T["max_bid_ask_pct"]:
                    continue

                # ROC = premium / capital secured
                capital_secured = strike * 100
                roc = (mark / strike) * 100 if strike > 0 else 0
                ann_yield = (roc / dte * 365) if dte > 0 else 0

                # Approximate delta from IV (Black-Scholes-ish)
                # yfinance doesn't provide greeks; use OTM % as proxy
                pct_otm = ((price - strike) / price) * 100

                entry = {
                    "opt_expiration": exp_str,
                    "opt_dte": dte,
                    "opt_strike": strike,
                    "opt_pct_otm": round(pct_otm, 2),
                    "opt_bid": bid,
                    "opt_ask": ask,
                    "opt_mark": round(mark, 2),
                    "opt_spread_pct": round(spread_pct, 2),
                    "opt_oi": int(oi),
                    "opt_iv": round(iv * 100, 2),  # as percentage
                    "opt_roc": round(roc, 2),
                    "opt_ann_yield": round(ann_yield, 2),
                    "opt_capital_secured": capital_secured,
                }

                # Pick the best by ROC among liquid strikes
                if best is None or entry["opt_roc"] > best["opt_roc"]:
                    best = entry

        return best

    except Exception as e:
        logger.warning(f"{ticker}: options chain error — {e}")
        return None


def fetch_earnings_proximity(ticker: str) -> int | None:
    """
    Return days until next earnings date.
    Returns None if unavailable.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None

        # yfinance returns calendar as dict or DataFrame depending on version
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                ed = cal["Earnings Date"].iloc[0]
            elif "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"].iloc[0]
            else:
                return None
        elif isinstance(cal, dict):
            dates = cal.get("Earnings Date")
            if not dates:
                return None
            ed = dates[0] if isinstance(dates, list) else dates
        else:
            return None

        if isinstance(ed, (datetime, pd.Timestamp)):
            return (ed.date() - datetime.now().date()).days
        return None

    except Exception as e:
        logger.debug(f"{ticker}: earnings date lookup failed — {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# SCORING ENGINE  (returns 0–100)
# ══════════════════════════════════════════════════════════════════════

def score_stock(d: dict, opts: dict | None = None,
                days_to_earnings: int | None = None) -> tuple[int, list[str], list[str]]:
    """
    Score a stock for Wheel-strategy suitability.
    Returns (score_pct, passes_list, flags_list).
    """
    T = THRESHOLDS
    score = 0
    max_score = 0
    passes: list[str] = []
    flags: list[str] = []

    def add(pts, max_pts, condition, label_pass, label_fail=None):
        nonlocal score, max_score
        max_score += max_pts
        if condition:
            score += pts
            passes.append(label_pass)
        elif label_fail:
            flags.append(label_fail)

    # ── HARD GATES ────────────────────────────────────────────────────
    if d["price"] < T["min_price"]:
        return 0, [], [f"✗ Price ${d['price']} below minimum ${T['min_price']}"]
    if d["mktcap_B"] < T["min_mktcap_B"]:
        return 0, [], [f"✗ Market cap ${d['mktcap_B']}B below ${T['min_mktcap_B']}B minimum"]

    # ── PILLAR 1: Earnings & Profitability (30 pts) ───────────────────
    add(8, 8,
        d["profit_margin"] >= T["profit_margin_min"],
        f"✓ Net margin {d['profit_margin']:.1f}% ≥ {T['profit_margin_min']}%",
        f"✗ Thin net margin {d['profit_margin']:.1f}%")

    add(8, 8,
        d["roe"] >= T["roe_min"],
        f"✓ ROE {d['roe']:.1f}% ≥ {T['roe_min']}%",
        f"✗ Weak ROE {d['roe']:.1f}%")

    add(4, 4,
        d["roe"] >= T["roe_good"],
        f"✓ Strong ROE {d['roe']:.1f}% ≥ {T['roe_good']}% (moat signal)")

    add(10, 10,
        d["fcf"] is not None and d["fcf"] > 0,
        "✓ Positive free cash flow",
        "✗ Negative or missing FCF")

    # ── PILLAR 2: Free Cash Flow Quality (10 pts) ─────────────────────
    _fy = d["fcf_yield"]  # may be None — evaluate labels AFTER the None check
    if _fy is not None:
        add(6, 6,
            _fy >= T["fcf_yield_min"],
            f"✓ FCF yield {_fy:.1f}% ≥ {T['fcf_yield_min']}%",
            f"✗ FCF yield {_fy:.1f}% (below {T['fcf_yield_min']}%)")
    else:
        max_score += 6
        flags.append("⚠ FCF yield N/A (no FCF or market cap data)")

    _om = d["operating_margin"] or 0.0
    add(4, 4,
        _om >= 10,
        f"✓ Operating margin {_om:.1f}%",
        f"✗ Operating margin {_om:.1f}%")

    # ── PILLAR 3: Balance Sheet / Debt (20 pts) ───────────────────────
    de = d["debt_equity"]
    if de is not None:
        add(8, 8,
            de <= T["debt_equity_max"],
            f"✓ D/E {de:.2f} ≤ {T['debt_equity_max']}",
            f"✗ High D/E {de:.2f}")
        add(5, 5,
            de <= T["debt_equity_good"],
            f"✓ Low D/E {de:.2f} ≤ {T['debt_equity_good']}")
    else:
        max_score += 13
        flags.append("⚠ Debt-to-equity ratio not available")

    _cr = d["current_ratio"]
    if _cr is not None:
        add(7, 7,
            _cr >= T["current_ratio_min"],
            f"✓ Current ratio {_cr:.2f} ≥ {T['current_ratio_min']}",
            f"✗ Current ratio {_cr:.2f} (below {T['current_ratio_min']})")
    else:
        max_score += 7
        flags.append("⚠ Current ratio N/A")

    # ── PILLAR 4: Valuation (25 pts) ──────────────────────────────────
    pe = d["pe"]
    if pe is not None and pe > 0:
        add(8, 8,
            pe <= T["pe_max"],
            f"✓ P/E {pe:.1f} ≤ {T['pe_max']}",
            f"✗ Pricey P/E {pe:.1f}")
        add(4, 4,
            pe <= T["pe_fair"],
            f"✓ Cheap P/E {pe:.1f} ≤ {T['pe_fair']}")
    else:
        max_score += 12

    pb = d["pb"]
    if pb is not None and pb > 0:
        add(5, 5,
            pb <= T["pb_max"],
            f"✓ P/B {pb:.1f} ≤ {T['pb_max']}",
            f"✗ High P/B {pb:.1f}")

    ev = d["ev_ebitda"]
    if ev is not None and ev > 0:
        add(8, 8,
            ev <= T["ev_ebitda_max"],
            f"✓ EV/EBITDA {ev:.1f} ≤ {T['ev_ebitda_max']}",
            f"✗ High EV/EBITDA {ev:.1f}")

    # ── PILLAR 5: Growth & Guidance (10 pts) ──────────────────────────
    _rg = d["revenue_growth"]
    if _rg is not None:
        add(4, 4,
            _rg >= T["revenue_growth_min"],
            f"✓ Revenue growth {_rg:.1f}%",
            f"✗ Revenue shrinking {_rg:.1f}%")
    else:
        max_score += 4
        flags.append("⚠ Revenue growth N/A")

    _eg = d["earnings_growth"]
    if _eg is not None:
        add(3, 3,
            _eg >= T["earnings_growth_min"],
            f"✓ Annual earnings growth {_eg:.1f}%",
            f"✗ Earnings declining {_eg:.1f}%")
    else:
        max_score += 3
        flags.append("⚠ Earnings growth N/A")

    _eqg = d["earnings_quarterly_growth"]
    if _eqg is not None:
        add(3, 3,
            _eqg >= 5,
            f"✓ Q/Q earnings growth {_eqg:.1f}% ≥ 5% (guidance proxy)",
            f"⚠ Q/Q earnings growth {_eqg:.1f}%")
    else:
        max_score += 3
        flags.append("⚠ Quarterly earnings growth N/A")

    # ── MOAT PROXIES ──────────────────────────────────────────────────
    add(3, 3,
        d["gross_margin"] >= 30,
        f"✓ Gross margin {d['gross_margin']:.1f}% ≥ 30% (pricing power)")

    add(2, 2,
        d["div_yield"] >= T["div_yield_bonus"],
        f"✓ Dividend {d['div_yield']:.1f}% (management confidence)")

    # ── ANALYST CONSENSUS BONUS ───────────────────────────────────────
    if d["analyst_rating"] is not None and (d["num_analysts"] or 0) >= 5:
        add(3, 3,
            d["analyst_rating"] <= 2.5,
            f"✓ Analyst consensus: Buy ({d['analyst_rating']:.1f}/5, n={d['num_analysts']})",
            f"⚠ Analyst rating {d['analyst_rating']:.1f}/5")

    if d["upside_pct"] is not None:
        add(2, 2,
            d["upside_pct"] >= 10,
            f"✓ Analyst upside {d['upside_pct']:.1f}%")

    # ── WHEEL-SPECIFIC: Beta sweet spot ───────────────────────────────
    if d["beta"] is not None:
        add(3, 3,
            0.4 <= d["beta"] <= 1.6,
            f"✓ Beta {d['beta']:.2f} (Wheel sweet spot 0.4–1.6)",
            f"⚠ Beta {d['beta']:.2f} outside sweet spot")

    # ── PILLAR 6: Options Chain Quality (20 pts) ──────────────────────
    if opts is not None:
        add(6, 6,
            opts["opt_oi"] >= T["min_open_interest"],
            f"✓ Open interest {opts['opt_oi']} ≥ {T['min_open_interest']}",
            f"✗ Low OI {opts['opt_oi']}")

        add(5, 5,
            opts["opt_spread_pct"] <= T["max_bid_ask_pct"],
            f"✓ Bid-ask spread {opts['opt_spread_pct']:.1f}% ≤ {T['max_bid_ask_pct']}%",
            f"✗ Wide spread {opts['opt_spread_pct']:.1f}%")

        add(5, 5,
            opts["opt_iv"] >= 25,
            f"✓ IV {opts['opt_iv']:.1f}% (premium-rich)",
            f"⚠ Low IV {opts['opt_iv']:.1f}% (thin premiums)")

        add(4, 4,
            opts["opt_roc"] >= 1.5,
            f"✓ ROC {opts['opt_roc']:.2f}% per cycle",
            f"⚠ ROC {opts['opt_roc']:.2f}% (low return on capital)")

        passes.append(
            f"  ↳ Best CSP: ${opts['opt_strike']} strike, "
            f"{opts['opt_pct_otm']:.1f}% OTM, "
            f"DTE {opts['opt_dte']}, "
            f"mark ${opts['opt_mark']:.2f}, "
            f"ann. yield {opts['opt_ann_yield']:.1f}%"
        )
    else:
        max_score += 20
        flags.append("✗ No liquid options chain in target DTE range")

    # ── Earnings proximity check ──────────────────────────────────────
    if days_to_earnings is not None and opts is not None:
        opt_dte = opts["opt_dte"]
        earnings_safe = days_to_earnings > opt_dte + T["min_earnings_dte"]
        add(3, 3,
            earnings_safe,
            f"✓ Earnings in {days_to_earnings}d — clear of {opt_dte}d expiration",
            f"✗ Earnings in {days_to_earnings}d — overlaps {opt_dte}d expiration!")
    elif days_to_earnings is not None:
        if days_to_earnings <= 30:
            flags.append(f"⚠ Earnings in {days_to_earnings}d — verify before selling CSP")
    else:
        # No penalty, just a note
        flags.append("⚠ Earnings date unknown — check manually")

    pct = round((score / max(max_score, 1)) * 100)
    return pct, passes, flags


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def grade(score: int) -> str:
    if score >= 80: return "A+"
    if score >= 70: return "A"
    if score >= 60: return "B+"
    if score >= 50: return "B"
    if score >= 40: return "C"
    return "D"


def wheel_tier(score: int) -> str:
    if score >= 75: return "🟢 STRONG"
    if score >= 60: return "🟡 SOLID"
    if score >= 45: return "🟠 WATCH"
    return "🔴 SKIP"


def _fmt(val, fmt=".1f", fallback="—") -> str:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return fallback
        return format(float(val), fmt)
    except Exception:
        return fallback


# ══════════════════════════════════════════════════════════════════════
# MAIN SCREENER
# ══════════════════════════════════════════════════════════════════════

def run_screener(
        include_canada: bool = True,
        include_usa: bool = True,
        min_score: int = 45,
        min_price: float = 10,
        max_price: float = 250,
        top_n: int = 40,
        verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═" * 70)
    print("   🎡  WHEEL STRATEGY STOCK SCREENER")
    print("   Canada (TSX) + USA  |  Fundamental Quality Filter")
    print("═" * 70)
    print(f"   Min Score: {min_score}%  |  Price: ${min_price}–${max_price}  |  Top N: {top_n}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 70 + "\n")

    universe: dict[str, str] = {}
    if include_canada:
        for grp, tickers in CANADIAN_STOCKS.items():
            for tk in tickers:
                universe[tk] = grp
    if include_usa:
        for grp, tickers in US_STOCKS.items():
            for tk in tickers:
                universe[tk] = grp

    total = len(universe)
    results = []
    skipped = 0

    print(f"  Fetching data for {total} tickers ...\n")

    for i, (ticker, group) in enumerate(universe.items(), 1):
        if verbose:
            pct = i / total
            bar = "█" * int(pct * 32) + "░" * (32 - int(pct * 32))
            print(f"\r  [{bar}] {i:>3}/{total}  {ticker:<14}", end="", flush=True)

        data = fetch_stock_data(ticker)
        if data is None:
            skipped += 1
            continue
        if not (min_price <= data["price"] <= max_price):
            skipped += 1
            continue

        # Fetch options chain and earnings proximity
        opts = fetch_options_data(ticker, data["price"])
        days_to_earn = fetch_earnings_proximity(ticker)

        sc, passes, flags = score_stock(data, opts=opts,
                                        days_to_earnings=days_to_earn)

        entry = {
            **data,
            "group": group,
            "score": sc,
            "grade": grade(sc),
            "tier": wheel_tier(sc),
            "passes": passes,
            "flags": flags,
            "days_to_earnings": days_to_earn,
        }
        # Merge options data into result row if available
        if opts:
            entry.update(opts)

        results.append(entry)

    print(f"\n\n  Done. Scored: {len(results)}  |  Skipped/Filtered: {skipped}\n")

    if not results:
        empty = pd.DataFrame()
        return empty, empty

    df_all = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    df_top = df_all[df_all["score"] >= min_score].head(top_n).reset_index(drop=True)

    return df_all, df_top


# ══════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════

def display_summary_table(df: pd.DataFrame):
    """Compact ranked table of top candidates."""
    print("\n" + "═" * 70)
    print("   📊  WHEEL STRATEGY CANDIDATES — RANKED TABLE")
    print("═" * 70)

    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["ticker"],
            r["name"][:22],
            r["tier"],
            f"{r['score']}%",
            f"${r['price']:.2f}",
            _fmt(r.get("pe")),
            f"{_fmt(r.get('roe'))}%",
            f"{_fmt(r.get('fcf_yield'))}%",
            _fmt(r.get("debt_equity")),
            f"${_fmt(r.get('opt_strike'))}" if r.get("opt_strike") else "—",
            _fmt(r.get("opt_dte"), fmt=".0f") if r.get("opt_dte") else "—",
            f"{_fmt(r.get('opt_iv'))}%" if r.get("opt_iv") else "—",
            f"{_fmt(r.get('opt_roc'))}%" if r.get("opt_roc") else "—",
            _fmt(r.get("opt_oi"), fmt=".0f") if r.get("opt_oi") else "—",
            f"{_fmt(r.get('opt_spread_pct'))}%" if r.get("opt_spread_pct") else "—",
            _fmt(r.get("days_to_earnings"), fmt=".0f") if r.get("days_to_earnings") else "—",
        ])

    headers = ["Ticker", "Name", "Tier", "Score", "Price",
               "P/E", "ROE", "FCF Yld",
               "D/E", "Strike", "DTE", "IV", "ROC", "OI", "Spread", "Earn"]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline",
                       numalign="right", stralign="left"))
    else:
        # Fallback plain print
        print("  " + "  ".join(f"{h:<12}" for h in headers))
        print("  " + "-" * 160)
        for row in rows:
            print("  " + "  ".join(f"{str(v):<12}" for v in row))
    print()


def display_by_tier(df: pd.DataFrame, n_each: int = 6):
    """Print top N per tier."""
    for tier_label in ["🟢 STRONG", "🟡 SOLID", "🟠 WATCH"]:
        sub = df[df["tier"] == tier_label]
        if sub.empty:
            continue
        print(f"\n  {'─' * 62}")
        print(f"  {tier_label}  (top {min(n_each, len(sub))})")
        print(f"  {'─' * 62}")
        for _, r in sub.head(n_each).iterrows():
            opt_info = ""
            if r.get("opt_strike"):
                opt_info = (f"  CSP:${r['opt_strike']:.0f}"
                            f" DTE:{r['opt_dte']:.0f}"
                            f" IV:{r.get('opt_iv', 0):.0f}%"
                            f" ROC:{r.get('opt_roc', 0):.1f}%")
            else:
                opt_info = "  ✗ No liquid opts"
            print(
                f"  {r['ticker']:<12}  Score:{r['score']:>3}%"
                f"  ${r['price']:>8.2f}"
                f"  P/E:{_fmt(r.get('pe')):>7}"
                f"  ROE:{_fmt(r.get('roe')):>6}%"
                f"{opt_info}"
            )


def display_market_split(df: pd.DataFrame):
    """Canada vs USA breakdown."""
    is_ca = df["ticker"].str.endswith(".TO") | df["ticker"].str.endswith(".V")
    ca = df[is_ca]
    us = df[~is_ca]

    for label, subset, flag in [("🍁 CANADIAN (TSX)", ca, "CA"), ("🦅 US", us, "US")]:
        print(f"\n  {'═' * 62}")
        print(f"  {label} — qualifying stocks")
        print(f"  {'═' * 62}")
        if subset.empty:
            print("  (none passed minimum score)")
            continue
        for _, r in subset.iterrows():
            print(
                f"  {r['ticker']:<12}  {r['tier']}  {r['score']:>3}%"
                f"  ${r['price']:>7.2f}  "
                f"P/E:{_fmt(r['pe']):>6}  "
                f"Div:{_fmt(r['div_yield']):>5}%  "
                f"{r['name'][:24]}"
            )


def display_scorecard(row: dict):
    """Detailed pass/fail card for one stock."""
    w = 64
    p, f = row.get("passes", []), row.get("flags", [])
    print(f"\n  ┌{'─' * w}┐")
    print(f"  │  {row['ticker']:<8} │ {row['name']:<28} │ {row['tier']:<12}  │")
    print(
        f"  │  Score: {row['score']}%  ({row['grade']})  │  ${row['price']} {row['currency']}  │  MCap: ${row['mktcap_B']}B  │")
    print(f"  ├{'─' * w}┤")
    for item in p:
        ln = f"  {item}"
        print(f"  │{ln:<{w}}│")
    if f:
        print(f"  │{'  ─── Flags ─── ':<{w}}│")
        for item in f:
            ln = f"  {item}"
            print(f"  │{ln:<{w}}│")
    print(f"  └{'─' * w}┘")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════

def save_results(df_top: pd.DataFrame, df_all: pd.DataFrame,
                 prefix: str = "wheel"):
    """Save top candidates and full screened universe to CSV + JSON."""
    csv_top = f"{prefix}_candidates.csv"
    csv_all = f"{prefix}_all_screened.csv"
    json_out = f"{prefix}_candidates.json"

    skip_cols = {"passes", "flags"}

    df_top[[c for c in df_top.columns if c not in skip_cols]].to_csv(csv_top, index=False)
    df_all[[c for c in df_all.columns if c not in skip_cols]].to_csv(csv_all, index=False)

    with open(json_out, "w") as fh:
        json.dump(df_top.to_dict(orient="records"), fh, indent=2, default=str)

    print(f"\n  💾 Saved: {csv_top}  ({len(df_top)} candidates)")
    print(f"  💾 Saved: {csv_all}  ({len(df_all)} total screened)")
    print(f"  💾 Saved: {json_out}  (full scorecards)")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    df_all, df_top = run_screener(
        include_canada=True,
        include_usa=True,
        min_score=45,  # ← lower to 35 to see more candidates
        min_price=10,  # ← CSP margin-friendly floor
        max_price=250,  # ← adjust to your account size
        top_n=40,
        verbose=True,
    )

    if df_top.empty:
        print("  ⚠ No stocks passed the minimum score. Try lowering min_score.")
    else:
        # ── Tables ────────────────────────────────────────────────────
        display_summary_table(df_top)
        display_by_tier(df_top, n_each=6)
        display_market_split(df_top)

        # ── Detailed scorecards for top 5 ─────────────────────────────
        print("\n\n" + "═" * 70)
        print("  🔍  DETAILED SCORECARD — TOP 5 WHEEL CANDIDATES")
        print("═" * 70)
        for _, row in df_top.head(5).iterrows():
            display_scorecard(row.to_dict())

        # ── Summary stats ─────────────────────────────────────────────
        is_ca = df_top["ticker"].str.endswith(".TO") | df_top["ticker"].str.endswith(".V")
        print("\n\n" + "═" * 70)
        print("  📈  SCREENING SUMMARY")
        print("═" * 70)
        for tier in ["🟢 STRONG", "🟡 SOLID", "🟠 WATCH"]:
            cnt = (df_top["tier"] == tier).sum()
            print(f"  {tier:<20}  {cnt:>3} stocks")
        print(f"\n  🍁 Canadian (TSX): {is_ca.sum()}  |  🦅 US: {(~is_ca).sum()}")
        print(f"  Avg Score:    {df_top['score'].mean():.1f}%")
        print(f"  Avg P/E:      {pd.to_numeric(df_top['pe'], errors='coerce').mean():.1f}")
        print(f"  Avg ROE:      {pd.to_numeric(df_top['roe'], errors='coerce').mean():.1f}%")
        print(f"  Avg FCF Yld:  {pd.to_numeric(df_top['fcf_yield'], errors='coerce').mean():.1f}%")

        # ── Strategy notes ────────────────────────────────────────────
        print("\n" + "═" * 70)
        print("  ⚙  WHEEL STRATEGY EXECUTION NOTES")
        print("═" * 70)
        print("""
  🟢 STRONG  → Ideal CSP candidates. Best strike, DTE, and ROC are
               shown in the scorecard. Sell on red days or IV rank > 30%.
               You genuinely want to own these on assignment.

  🟡 SOLID   → Sell CSPs around high-IV events (earnings, macro).
               Set your cost basis target before opening the trade.

  🟠 WATCH   → Track for improving fundamentals. Paper-trade first.

  GENERAL RULES:
  • Screener verifies OI, bid-ask spread, IV, and earnings proximity
  • Ideal Wheel price range: $15–$100 (keeps margin requirements low)
  • For Canadian stocks: many TSX names also trade on NYSE (check both)
  • Never Wheel a stock you wouldn't hold through a 30% drawdown
  • Roll CSPs out in time (same strike) before taking assignment if
    IV has spiked — you capture extra premium without being forced in
  • Close at 50% max profit and redeploy for faster capital rotation
        """)

        # ── Save ──────────────────────────────────────────────────────
        save_results(df_top, df_all)

    print("\n  ✅  Screener complete.\n")
