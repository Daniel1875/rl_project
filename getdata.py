"""
Download NASDAQ TotalView-ITCH MBP-10 data from Databento.

Matches the data specification of Spooner et al. (2018), Section 3:
  - 8 months of data (January–August)
  - Top-5 order book depth (bid/ask prices + sizes at each level)
  - Transaction records (action, side, price, size, timestamp)

Each agent is trained on one security only.  This script downloads BIIB.

MBP-10 columns kept per record:
  ts_event              nanosecond UTC timestamp of the event
  action                event type: A=add, C=cancel, T=trade, F=fill, D=delete
  side                  order side: A=ask, B=bid, N=none (trades)
  price                 event price (Databento fixed-point: divide by 1e9 for USD)
  size                  event size in shares
  bid_px_00..bid_px_04  top-5 bid prices after this event
  bid_sz_00..bid_sz_04  top-5 bid sizes
  ask_px_00..ask_px_04  top-5 ask prices
  ask_sz_00..ask_sz_04  top-5 ask sizes
"""

import os
import databento as db

SYMBOL = "BIIB"

START = "2019-01-01"
END   = "2019-08-31"

OUTPUT_DIR = "data"
API_KEY    = ""

LEVEL_COLS = [
    col
    for lvl in range(5)
    for col in (
        f"bid_px_{lvl:02d}", f"bid_sz_{lvl:02d}",
        f"ask_px_{lvl:02d}", f"ask_sz_{lvl:02d}",
    )
]
KEEP_COLS = ["ts_event", "symbol", "action", "side", "price", "size"] + LEVEL_COLS


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}.parquet")

if os.path.exists(out_path):
    print(f"{SYMBOL} already downloaded at {out_path}")
    raise SystemExit(0)

client = db.Historical(API_KEY)

nbytes = client.metadata.get_billable_size(
    dataset="XNAS.ITCH", symbols=[SYMBOL], schema="mbp-10", start=START, end=END,
)
cost = client.metadata.get_cost(
    dataset="XNAS.ITCH", symbols=[SYMBOL], schema="mbp-10", start=START, end=END,
)
print(f"Symbol : {SYMBOL}")
print(f"Period : {START} to {END}")
print(f"Size   : {fmt_bytes(nbytes)}")
print(f"Cost   : ${cost:.2f}\n")

answer = input("Proceed with download? [y/N] ").strip().lower()
if answer != "y":
    print("Aborted.")
    raise SystemExit(0)

print(f"\nFetching {SYMBOL} ...", end=" ", flush=True)
data = client.timeseries.get_range(
    dataset="XNAS.ITCH",
    symbols=[SYMBOL],
    schema="mbp-10",
    stype_in="raw_symbol",
    start=START,
    end=END,
)
df = data.to_df()
df = df[[c for c in KEEP_COLS if c in df.columns]]
df.to_parquet(out_path, index=False)
print(f"{len(df):,} rows  →  {out_path}")
