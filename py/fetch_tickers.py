import time

import yfinance as yf
from yfinance import EquityQuery

q = EquityQuery("eq", ["region", "ca"])
all_tickers = []
offset = 0

while True:
    try:
        result = yf.screen(q, offset=offset, size=250)
        quotes = result.get("quotes", [])
        if not quotes:
            break
        all_tickers.extend([qt["symbol"] for qt in quotes])
        offset += 250
        print(f"Fetched {len(all_tickers)} so far...")
        time.sleep(1)
    except Exception as e:
        print(f"Error at offset {offset}: {e}")
        break

all_tickers = sorted(set(all_tickers))
print(f"\nTotal unique Canadian tickers: {len(all_tickers)}")

with open("../data/can_tickers_full", "w") as f:
    for t in all_tickers:
        f.write(t + "\n")

print("Saved to data/can_tickers_full")
