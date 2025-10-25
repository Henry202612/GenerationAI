import os
import re
import glob
import json
import math
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# =========================
# Paths and Parameters (aligned with the original project)
# =========================
OUTPUT_BASE = "data_saving"
PRICE_DIR = os.path.join(OUTPUT_BASE, "price_data")
TRADING_DATES_FILE = os.path.join(OUTPUT_BASE, "all_trading_dates.csv")

# Model/Data Parameters
LOOKBACK   = 60               # Window length L
FEAT_DIM   = 5                # [log_ret, log_vol, log_amt, to_rate, log_mcap]
STATIC_DIM = 1                # [log_circulating_shares]
VALID_DAYS = 30               # Validation set: the most recent N trading days before the prediction start date
PRED_START = pd.Timestamp("2025-10-09")  # Synchronized with later tasks
PRED_END   = pd.Timestamp("2025-11-09")

# TFRecord Shard Control (adjustable based on machine capabilities)
MAX_EXAMPLES_PER_SHARD = 12000   # Maximum samples per shard
TFRECORD_DIR = os.path.join(OUTPUT_BASE, "tfrecord")
TRAIN_DIR = os.path.join(TFRECORD_DIR, "train")
VALID_DIR = os.path.join(TFRECORD_DIR, "valid")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)

SEED = 20251021
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Utility Functions
# =========================
def log(msg: str):
    print(msg, flush=True)

def read_csv_auto(fp: str) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    for enc in encs:
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(fp)  # Fallback

def ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard columns: date, open, high, low, close, volume, amount, circulating_shares, market_cap, turnover_rate
    """
    lower = {c: str(c).strip().lower() for c in df.columns}
    rev = {v: k for k, v in lower.items()}
    rename_map = {}
    alias = {
        "date": ["date", "日期", "time", "交易日期"],
        "open": ["open", "开盘", "openprice"],
        "high": ["high", "最高", "highprice"],
        "low": ["low", "最低", "lowprice"],
        "close": ["close", "收盘", "closeprice", "adjclose", "adj_close"],
        "volume": ["volume", "成交量", "vol"],
        "amount": ["amount", "成交额", "amt", "turnover"],
        "circulating_shares": ["circulating_shares", "流通股本", "flow_shares"],
        "market_cap": ["market_cap", "市值", "总市值", "流通市值"],
        "turnover_rate": ["turnover_rate", "换手率", "turnover"]
    }
    for std, names in alias.items():
        for n in names:
            if n in lower.values():
                rename_map[rev[n]] = std
                break
    df = df.rename(columns=rename_map)
    for c in ["date", "open", "high", "low", "close", "volume", "amount",
              "circulating_shares", "market_cap", "turnover_rate"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df[["date", "open", "high", "low", "close", "volume", "amount",
             "circulating_shares", "market_cap", "turnover_rate"]].copy()

    # downcast + date parsing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    num_cols = ["open", "high", "low", "close", "volume", "amount",
                "circulating_shares", "market_cap", "turnover_rate"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate feature columns and static features stored in df.attrs["static_feat"]
    """
    df = df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["log_vol"] = np.log1p(df["volume"].clip(lower=0))
    df["log_amt"] = np.log1p(df["amount"].clip(lower=0))
    df["to_rate"] = df["turnover_rate"].astype(float)
    df["log_mcap"] = np.log1p(df["market_cap"].clip(lower=0)) if "market_cap" in df.columns else np.nan

    df[["log_ret", "log_vol", "log_amt", "to_rate", "log_mcap"]] = \
        df[["log_ret", "log_vol", "log_amt", "to_rate", "log_mcap"]].fillna(method="ffill").fillna(0.0)

    circ = df["circulating_shares"].ffill().dropna()
    static = float(np.log1p(circ.iloc[-1])) if not circ.empty else 0.0
    df.attrs["static_feat"] = np.array([static], dtype=np.float32)

    df["target_next_logret"] = df["log_ret"].shift(-1)
    return df

def build_sequences_stream(df: pd.DataFrame, lookback: int):
    """
    Generator: yields (X_seq[L,F], y[1]) without accumulating memory.
    """
    feat_cols = ["log_ret", "log_vol", "log_amt", "to_rate", "log_mcap"]
    feats = df[feat_cols].values.astype(np.float32)
    y = df["target_next_logret"].values.astype(np.float32)
    N = len(df)
    if N < lookback + 2:
        return
    for i in range(N - lookback - 1):
        xs = feats[i:i + lookback, :]     # [L, F]
        yy = y[i + lookback]              # scalar
        yield xs, yy

def list_price_files(price_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(price_dir, "*.csv")))

def extract_code6_from_path(path: str) -> Optional[str]:
    m = re.search(r"(\d{6})", os.path.basename(path))
    return m.group(1) if m else None

# =========================
# TFRecord Write Helper
# =========================
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature_list(values: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values.reshape(-1).tolist()))

def example_from_sample(x_seq_f16: np.ndarray,  # [L,F] float16
                        y_f32: float,
                        static_f16: np.ndarray,  # [S] float16
                        shape_seq: Tuple[int, int],
                        shape_static: Tuple[int]):
    """
    Store as bytes + shape, more disk and memory friendly.
    """
    L, F = shape_seq
    (S,) = shape_static
    feature = {
        "x_seq":      _bytes_feature(x_seq_f16.tobytes()),   # float16 bytes
        "x_seq_L":    tf.train.Feature(int64_list=tf.train.Int64List(value=[L])),
        "x_seq_F":    tf.train.Feature(int64_list=tf.train.Int64List(value=[F])),
        "x_static":   _bytes_feature(static_f16.tobytes()),  # float16 bytes
        "x_static_S": tf.train.Feature(int64_list=tf.train.Int64List(value=[S])),
        "y":          tf.train.Feature(float_list=tf.train.FloatList(value=[float(y_f32)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

class ShardedTFWriter:
    """
    Simple sharded writer: move to next file after reaching MAX_EXAMPLES_PER_SHARD
    """
    def __init__(self, out_dir: str, prefix: str, max_examples_per_shard: int = 12000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_per = max_examples_per_shard
        self.count_in_shard = 0
        self.total = 0
        self.shard_idx = 0
        self.writer = None
        self._open_new()

    def _open_new(self):
        if self.writer:
            self.writer.close()
        fname = f"{self.prefix}-{self.shard_idx:05d}.tfrecord.gz"
        path = os.path.join(self.out_dir, fname)
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        self.writer = tf.io.TFRecordWriter(path, options=options)
        self.count_in_shard = 0
        log(f"[OPEN] {path}")

    def write(self, example: tf.train.Example):
        if self.count_in_shard >= self.max_per:
            self.shard_idx += 1
            self._open_new()
        self.writer.write(example.SerializeToString())
        self.count_in_shard += 1
        self.total += 1

    def close(self):
        if self.writer:
            self.writer.close()
            log(f"[CLOSE] shard {self.shard_idx:05d} with {self.count_in_shard} examples")

# =========================
# Main Flow: Build train/valid TFRecords
# =========================
def main():
    assert os.path.isdir(PRICE_DIR), f"Price-volume data directory not found: {PRICE_DIR}"
    assert os.path.isfile(TRADING_DATES_FILE), f"Missing trading date file: {TRADING_DATES_FILE}"

    # Trading days (for time slicing)
    trade_days = pd.read_csv(TRADING_DATES_FILE, header=None).iloc[:, 0]
    trade_days = pd.to_datetime(trade_days, errors="coerce").dropna().dt.normalize().unique()
    trade_days = pd.DatetimeIndex(sorted(pd.to_datetime(trade_days)))

    pred_start_prev = trade_days[trade_days < PRED_START.normalize()]
    if len(pred_start_prev) < VALID_DAYS + LOOKBACK + 5:
        log("[WARN] Historical trading days are short, validation set may be small.")
    valid_start_day = pred_start_prev[-VALID_DAYS] if len(pred_start_prev) >= VALID_DAYS else pred_start_prev[0]
    train_end_day = valid_start_day - pd.Timedelta(days=1)
    valid_end_day = pred_start_prev[-1]  # Validation set upper bound (the day before the prediction start date)

    log(f"[TIME CUT] train <= {train_end_day.date()} ; valid <= {valid_end_day.date()} (cover last {VALID_DAYS} days)")

    # Shard Writers
    train_writer = ShardedTFWriter(TRAIN_DIR, prefix="shard", max_examples_per_shard=MAX_EXAMPLES_PER_SHARD)
    valid_writer = ShardedTFWriter(VALID_DIR, prefix="shard", max_examples_per_shard=MAX_EXAMPLES_PER_SHARD)

    # Stats
    stats = {
        "lookback": LOOKBACK,
        "feat_dim": FEAT_DIM,
        "static_dim": STATIC_DIM,
        "valid_days": VALID_DAYS,
        "pred_start": str(PRED_START.date()),
        "pred_end": str(PRED_END.date()),
        "train_examples": 0,
        "valid_examples": 0,
        "train_shards": 0,
        "valid_shards": 0,
        "price_dir": PRICE_DIR,
        "trading_dates_file": TRADING_DATES_FILE
    }

    files = list_price_files(PRICE_DIR)
    if not files:
        raise RuntimeError("No CSV files found in the price directory.")

    for i, fp in enumerate(files, 1):
        try:
            df = ensure_price_columns(read_csv_auto(fp))
            if df.shape[0] < LOOKBACK + 2:
                continue

            # -------- A) Train Samples --------
            df_train = df[df["date"] <= train_end_day]
            if df_train.shape[0] >= LOOKBACK + 2:
                df_train = compute_features(df_train)
                static_vec = df_train.attrs["static_feat"].astype(np.float32)  # [1]
                static_f16 = static_vec.astype(np.float16)

                for x_seq, y in build_sequences_stream(df_train, LOOKBACK):
                    # Write to float16
                    ex = example_from_sample(
                        x_seq_f16=x_seq.astype(np.float16),
                        y_f32=float(y),
                        static_f16=static_f16,
                        shape_seq=(LOOKBACK, FEAT_DIM),
                        shape_static=(STATIC_DIM,)
                    )
                    train_writer.write(ex)
                stats["train_examples"] = train_writer.total

            # -------- B) Validation Samples --------
            df_valid = df[df["date"] <= valid_end_day]
            if df_valid.shape[0] >= LOOKBACK + 2:
                df_valid = compute_features(df_valid)
                static_vec_v = df_valid.attrs["static_feat"].astype(np.float32)
                static_f16_v = static_vec_v.astype(np.float16)

                for x_seq, y in build_sequences_stream(df_valid, LOOKBACK):
                    ex = example_from_sample(
                        x_seq_f16=x_seq.astype(np.float16),
                        y_f32=float(y),
                        static_f16=static_f16_v,
                        shape_seq=(LOOKBACK, FEAT_DIM),
                        shape_static=(STATIC_DIM,)
                    )
                    valid_writer.write(ex)
                stats["valid_examples"] = valid_writer.total

            if i % 500 == 0:
                log(f"[PROGRESS] {i}/{len(files)} files processed | train={train_writer.total} valid={valid_writer.total}")

        except Exception as e:
            log(f"[WARN] Processing failed for {os.path.basename(fp)}: {e}")

    # Close writers and record final shard count
    train_writer.close()
    valid_writer.close()
    stats["train_shards"] = train_writer.shard_idx + 1
    stats["valid_shards"] = valid_writer.shard_idx + 1

    # Write manifest
    manifest_path = os.path.join(TFRECORD_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"[DONE] Preprocessing complete. Manifest written to: {manifest_path}")
    log(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
