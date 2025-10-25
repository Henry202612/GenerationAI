import os
import re
import glob
import math
import time
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd

# Optional: Enable AKShare to automatically fetch circulating shares
USE_AKSHARE = True
if USE_AKSHARE:
    try:
        import akshare as ak
    except Exception as e:
        print("[WARN] AKShare not installed, will skip fetching circulating shares:", e)
        USE_AKSHARE = False

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =========================
# 0) Path Configuration (Modify as needed)
# =========================
# Raw price data directory: e.g., SH#600000.csv / SZ#000001.csv / BJ#xxxxxx.csv
PRICE_RAW_DIR = "data/A_share_stock_price_data"
# Raw financial data directory: e.g., 600000.资产负债表-*.csv / or 600000.*.csv
FINANCIAL_RAW_DIR = "data/Financial_report_data"

# Processed dataset root directory
OUTPUT_BASE = "data_saving"
OUTPUT_PRICE_DIR = os.path.join(OUTPUT_BASE, "price_data")
OUTPUT_FIN_DIR   = os.path.join(OUTPUT_BASE, "financial_data")
TRADING_DATES_FILE = os.path.join(OUTPUT_BASE, "all_trading_dates.csv")

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(OUTPUT_PRICE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIN_DIR, exist_ok=True)

# Start date for price data (can be modified)
START_DATE = pd.Timestamp("2015-01-01")

# =========================
# 1) Utility Functions
# =========================
def read_csv_auto(fp: str, header_maybe: bool = True) -> pd.DataFrame:
    """
    Try reading the CSV file with various encodings and header formats.
    If header_maybe=True, it first attempts to read with headers, otherwise reads without headers.
    """
    encs = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    errs = []
    if header_maybe:
        for enc in encs:
            try:
                df = pd.read_csv(fp, encoding=enc)
                return df
            except Exception as e:
                errs.append((enc, "with_header", str(e)))
        for enc in encs:
            try:
                df = pd.read_csv(fp, encoding=enc, header=None)
                return df
            except Exception as e:
                errs.append((enc, "no_header", str(e)))
        # Finally try the default
        try:
            return pd.read_csv(fp)
        except Exception as e:
            errs.append(("default", "final", str(e)))
            raise RuntimeError(f"[read_csv_auto] Failed to read: {fp}\n{errs}")
    else:
        for enc in encs:
            try:
                df = pd.read_csv(fp, encoding=enc, header=None)
                return df
            except Exception as e:
                errs.append((enc, "no_header", str(e)))
        try:
            return pd.read_csv(fp, header=None)
        except Exception as e:
            errs.append(("default", "final", str(e)))
            raise RuntimeError(f"[read_csv_auto] Failed to read: {fp}\n{errs}")

def normalize_code_from_filename(filename: str) -> Optional[str]:
    """
    Extract the 6-digit stock code from a filename like 'SH#600006.csv' or 'SZ#000001.csv'.
    """
    base = os.path.basename(filename)
    m = re.search(r"#(\d{6})", base)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})", base)
    return m.group(1) if m else None

def ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the price data to standard columns:
    ["date","open","high","low","close","volume","amount"]
    Compatible with no header, different positions, or different capitalization.
    """
    cols = [c.lower() for c in df.columns.astype(str).tolist()]
    # If it is obviously without a header (common 7 columns)
    if len(df.columns) >= 7 and not any(x in cols for x in ["date", "日期"]):
        df = df.iloc[:, :7].copy()
        df.columns = ["date", "open", "high", "low", "close", "volume", "amount"]
        return df

    # Attempt mapping
    mapping = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["date", "日期", "time", "交易日期"]:
            mapping[c] = "date"
        elif cl in ["open", "开盘", "openprice"]:
            mapping[c] = "open"
        elif cl in ["high", "最高", "highprice"]:
            mapping[c] = "high"
        elif cl in ["low", "最低", "lowprice"]:
            mapping[c] = "low"
        elif cl in ["close", "收盘", "closeprice", "adjclose", "adj_close"]:
            mapping[c] = "close"
        elif cl in ["volume", "成交量", "vol"]:
            mapping[c] = "volume"
        elif cl in ["amount", "成交额", "amt", "turnover"]:
            mapping[c] = "amount"
    df = df.rename(columns=mapping)

    # If columns are still missing, fill in by position
    need = ["date", "open", "high", "low", "close", "volume", "amount"]
    if not all(c in df.columns for c in need):
        base = df.copy()
        # Only take the first 7 columns as fallback
        if base.shape[1] >= 7:
            base = base.iloc[:, :7]
            base.columns = need
            df = base
        else:
            raise ValueError("Insufficient columns in price data, cannot align to 7 standard columns")
    return df[need].copy()

def detect_all_trading_dates(price_files: List[str]) -> pd.DatetimeIndex:
    """
    Gather all trading dates based on the "full" stock CSV files (no sampling).
    """
    all_dates = []
    for i, fp in enumerate(price_files, 1):
        try:
            df0 = read_csv_auto(fp, header_maybe=True)
            df0 = ensure_price_columns(df0)
            dt = pd.to_datetime(df0["date"], errors="coerce").dropna().dt.normalize().unique()
            all_dates.extend(dt)
        except Exception as e:
            print(f"[WARN] Failed to extract trading dates from {os.path.basename(fp)}: {e}")
        if i % 500 == 0:
            print(f"  Processed {i}/{len(price_files)} files for trading dates...")
    if not all_dates:
        raise RuntimeError("No trading dates extracted from any stock CSV")
    uniq = sorted(pd.unique(pd.to_datetime(all_dates)))
    return pd.DatetimeIndex(uniq)

def get_float_shares_from_akshare(code6: str) -> Optional[float]:
    """
    Fetch the latest circulating shares from AKShare (in shares).
    """
    if not USE_AKSHARE:
        return np.nan
    try:
        df = ak.stock_value_em(symbol=code6)
        time.sleep(1)  # Slight delay to avoid IP blocking
        latest = df.iloc[-1]
        shares = latest.get("流通股本", np.nan)
        return float(shares) if pd.notna(shares) else np.nan
    except Exception as e:
        print(f"[akshare] Failed to fetch circulating shares for {code6}: {e}")
        return np.nan

def format_yyyymmdd_to_ymd(x):
    """
    Convert 'YYYYMMDD' -> 'YYYY/M/D'; return original if parsing fails.
    """
    s = str(x)
    if len(s) == 8 and s.isdigit():
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if pd.notna(dt):
            return f"{dt.year}/{dt.month}/{dt.day}"
    return x

# =========================
# 2) Main Process
# =========================
def main():
    print("=== Full Process Started ===")
    # 2.1 Scan all raw price CSV files
    price_files = glob.glob(os.path.join(PRICE_RAW_DIR, "*#*.csv"))
    price_files += glob.glob(os.path.join(PRICE_RAW_DIR, "*.csv"))  # Fallback
    price_files = sorted(list(set(price_files)))
    if not price_files:
        raise RuntimeError(f"No CSV files found in raw price directory: {PRICE_RAW_DIR}")
    print(f"[INFO] Found {len(price_files)} stock CSV files")

    # 2.2 Generate all market trading dates (no sampling)
    print("[STEP] Gathering all market trading dates ...")
    all_trading_dates = detect_all_trading_dates(price_files)
    # Optional: Only keep dates after START_DATE
    all_trading_dates = all_trading_dates[all_trading_dates >= START_DATE.normalize()]
    pd.DataFrame(all_trading_dates).to_csv(TRADING_DATES_FILE, index=False, header=False, encoding="utf-8")
    print(f"[SAVE] all_trading_dates.csv -> {TRADING_DATES_FILE} ({len(all_trading_dates)} days)")

    # 2.3 Process price data for each stock (fill in suspended days, interpolation, calculate fields)
    print("[STEP] Processing price data for each stock ...")
    for i, fp in enumerate(price_files, 1):
        try:
            code6 = normalize_code_from_filename(fp)
            if not code6:
                print(f"[WARN] Unable to recognize stock code, skipping: {os.path.basename(fp)}")
                continue

            df0 = read_csv_auto(fp, header_maybe=True)
            df0 = ensure_price_columns(df0)

            # Normalize and filter data
            df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
            df0 = df0.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            # Filter by start date
            df0 = df0[df0["date"] >= START_DATE].reset_index(drop=True)
            if df0.empty:
                print(f"[SKIP] No data for {code6} since {START_DATE.date()}")
                continue

            ipo_date  = df0["date"].min().normalize()
            last_date = df0["date"].max().normalize()
            # Select valid trading days within the stock's trading period
            mask = (all_trading_dates >= ipo_date) & (all_trading_dates <= last_date)
            target_dates = pd.DatetimeIndex(all_trading_dates[mask])

            # Align and fill data
            base = df0.set_index(df0["date"].dt.normalize()).drop(columns=["date"])
            base = base.reindex(target_dates)

            # Interpolate prices (only close price, others generally don't interpolate)
            base["close"] = pd.to_numeric(base["close"], errors="coerce")
            base["close"] = base["close"].interpolate(method="time")
            base["close"] = base["close"].round(2)

            # Convert other columns to numeric
            for col in ["open", "high", "low", "volume", "amount"]:
                if col in base.columns:
                    base[col] = pd.to_numeric(base[col], errors="coerce")

            out_df = base.reset_index().rename(columns={"index": "date"})
            # Get circulating shares
            shares_val = get_float_shares_from_akshare(code6) if USE_AKSHARE else np.nan
            if pd.isna(shares_val):
                print(f"[TIP] Circulating shares for {code6} not found (AKShare), column will be NaN")
            out_df["circulating_shares"] = float(shares_val) if pd.notna(shares_val) else np.nan

            # Calculate market cap and turnover rate
            out_df["market_cap"] = out_df["close"] * out_df["circulating_shares"]
            out_df["turnover_rate"] = np.where(
                pd.notna(out_df["circulating_shares"]) & (out_df["circulating_shares"] > 0),
                (out_df["volume"] / out_df["circulating_shares"] * 100.0),
                np.nan
            )
            out_df["market_cap"] = out_df["market_cap"].round(0)
            out_df["turnover_rate"] = out_df["turnover_rate"].round(2)

            # Standard output columns
            cols_out = [
                "date", "open", "high", "low", "close", "volume", "amount",
                "circulating_shares", "market_cap", "turnover_rate"
            ]
            for c in cols_out:
                if c not in out_df.columns:
                    out_df[c] = np.nan
            out_df = out_df[cols_out]

            # Save output
            out_name = f"{os.path.basename(fp)}"  # Keep original filename (with prefix)
            out_path = os.path.join(OUTPUT_PRICE_DIR, out_name)
            out_df.to_csv(out_path, index=False, encoding="utf-8")
            if i % 200 == 0:
                print(f"  Processed {i}/{len(price_files)}: {code6}")
        except Exception as e:
            print(f"[ERROR] Failed to process {os.path.basename(fp)}: {e}")

    print(f"[DONE] Price data saved to: {OUTPUT_PRICE_DIR}")

    # 2.4 Process financial CSV files: No filter for stock code, process all files directly
    print("[STEP] Processing financial data ...")
    fin_files = glob.glob(os.path.join(FINANCIAL_RAW_DIR, "*.csv"))
    if not fin_files:
        print(f"[WARN] No CSV files found in financial directory: {FINANCIAL_RAW_DIR}")
    for j, f in enumerate(fin_files, 1):
        try:
            df = read_csv_auto(f, header_maybe=True)
            # Keep only rows where the 4th column (index 3) is > 20140930 (if it exists and is formatted like YYYYMMDD)
            if df.shape[1] > 3:
                ser = df.iloc[:, 3].astype(str)
                mask = ser.str.fullmatch(r"\d{8}") & (ser.astype(int) > 20140930)
                df = df[mask].copy()

            # Normalize the 2nd and 3rd columns' date formats (if they exist and are in YYYYMMDD)
            for col_idx in [1, 2]:
                if df.shape[1] > col_idx:
                    df.iloc[:, col_idx] = df.iloc[:, col_idx].map(format_yyyymmdd_to_ymd)

            out_fin = os.path.join(OUTPUT_FIN_DIR, os.path.basename(f))
            df.to_csv(out_fin, index=False, encoding="utf-8")
            if j % 500 == 0:
                print(f"  Processed {j}/{len(fin_files)} financial files")
        except Exception as e:
            print(f"[ERROR] Failed to process financial file {os.path.basename(f)}: {e}")
    print(f"[DONE] Financial data saved to: {OUTPUT_FIN_DIR}")

    print("=== Full Process Completed ===")
    print("Output directories:")
    print("  Price data ->", OUTPUT_PRICE_DIR)
    print("  Financial data ->", OUTPUT_FIN_DIR)
    print("  Trading dates file ->", TRADING_DATES_FILE)

if __name__ == "__main__":
    main()
