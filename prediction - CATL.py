import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# pip install akshare==1.* pandas numpy tensorflow==2.* scikit-learn
import akshare as ak
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# =========================
# Basic configuration
# =========================
OUTPUT_BASE = "data_saving"  # Change to a purely English path
MODELS_DIR  = os.path.join(OUTPUT_BASE, "models_pso")

LOOKBACK   = 60
FEAT_DIM   = 5   # open, high, low, close, volume
STATIC_DIM = 1   # Constant placeholder

FINAL_BATCH_SIZE = 64
FINAL_EPOCHS     = 80
SEED = 20251021
USE_MIXED_PRECISION = True

tf.keras.utils.set_random_seed(SEED)
if USE_MIXED_PRECISION:
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled.")
    except Exception as e:
        print("[WARN] Mixed precision not enabled:", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print(f"[INFO] GPUs detected: {len(gpus)} -> {gpus}")
    else:
        print("[WARN] No GPU detected. Will run on CPU.")
except Exception as e:
    print(f"[WARN] GPU init issue: {e}")

# =========================
# Model Definition (Same as the original version)
# =========================
class LSTMMLP_PSO(tf.keras.Model):
    def __init__(self,
                 lookback=LOOKBACK, in_dim=FEAT_DIM, static_dim=STATIC_DIM,
                 lstm_units=64, dense1=64, dense2=32, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.in_dim = in_dim
        self.static_dim = static_dim

        self.mask = tf.keras.layers.Masking(mask_value=0.0)
        self.lstm = tf.keras.layers.LSTM(
            int(lstm_units), return_sequences=False,
            activation="tanh", recurrent_activation="sigmoid",
            use_bias=True, recurrent_dropout=0.0, unroll=False,
            dtype="float32"
        )
        self.post_rnn_bn = tf.keras.layers.BatchNormalization(dtype="float32")
        self.concat = tf.keras.layers.Concatenate()
        self.pre_mlp_bn = tf.keras.layers.BatchNormalization(dtype="float32")
        self.mlp_dense1 = tf.keras.layers.Dense(int(dense1), activation="relu")
        self.do1 = tf.keras.layers.Dropout(float(dropout)) if dropout > 0 else None
        self.mlp_dense2 = tf.keras.layers.Dense(int(dense2), activation="relu")
        self.do2 = tf.keras.layers.Dropout(float(dropout)) if dropout > 0 else None
        self.out = tf.keras.layers.Dense(1, dtype="float32")

    def call(self, inputs, training=False):
        x_seq = inputs["x_seq"]
        x_static = inputs["x_static"]
        x_seq.set_shape([None, self.lookback, self.in_dim])
        x_static.set_shape([None, STATIC_DIM])

        x = self.mask(x_seq)
        lstm_out = self.lstm(x, training=training)
        lstm_out = self.post_rnn_bn(lstm_out, training=training)

        h = self.concat([lstm_out, x_static])
        h = self.pre_mlp_bn(h, training=training)
        h = self.mlp_dense1(h, training=training)
        if self.do1 is not None:
            h = self.do1(h, training=training)
        h = self.mlp_dense2(h, training=training)
        if self.do2 is not None:
            h = self.do2(h, training=training)
        out = self.out(h, training=training)  # Predict the standardized "residual"
        return out

def build_model_with_hparams(hp: dict):
    model = LSTMMLP_PSO(
        lstm_units=hp.get("lstm_units", 64),
        dense1=hp.get("dense1", 64),
        dense2=hp.get("dense2", 32),
        dropout=hp.get("dropout", 0.0),
    )
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
    ]
    base_lr = float(hp.get("lr", 1e-3))
    if USE_MIXED_PRECISION:
        base_lr = min(base_lr, 1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, run_eagerly=True)  # Enable eager execution for debugging
    return model

def find_latest_trained_model_dir(models_root: str) -> str:
    if not os.path.isdir(models_root):
        return None
    subs = [os.path.join(models_root, d) for d in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, d))]
    if not subs:
        return None
    subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subs[0]

def try_load_meta_and_model(latest_dir: str):
    hp = {"lstm_units": 64, "dense1": 64, "dense2": 32, "dropout": 0.0, "lr": 1e-3}
    loaded_from = None
    if latest_dir and os.path.isdir(latest_dir):
        meta_path = os.path.join(latest_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "best_hparams" in meta:
                    hp.update(meta["best_hparams"])
                print("[INFO] loaded best_hparams from meta.json:", hp)
            except Exception as e:
                print("[WARN] read meta.json failed:", e)
        try:
            model = build_model_with_hparams(hp)
            dummy = {"x_seq": tf.zeros([1, LOOKBACK, FEAT_DIM], dtype=tf.float32),
                     "x_static": tf.zeros([1, STATIC_DIM], dtype=tf.float32)}
            _ = model(dummy, training=False)
            pick = None
            if os.path.isfile(os.path.join(latest_dir, "best.weights.h5")):
                pick = os.path.join(latest_dir, "best.weights.h5")
            elif os.path.isfile(os.path.join(latest_dir, "final.weights.h5")):
                pick = os.path.join(latest_dir, "final.weights.h5")
            if pick:
                model.load_weights(pick)
                loaded_from = pick
                print(f"[LOAD] weights loaded from {pick}")
                return model, hp, pick
        except Exception as e:
            print("[WARN] build/load weights failed:", e)
    print("[INFO] fallback: build a fresh model with default/best_hparams.")
    model = build_model_with_hparams(hp)
    return model, hp, None

# =========================
# Data fetching and "residual target" sample generation
# =========================
def fetch_cn1088_daily(start="20200101", end="20250829", adjust="qfq"):
    df = ak.stock_zh_a_hist(symbol="300750", period="daily",
                            start_date=start, end_date=end, adjust=adjust)
    col_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
               "收盘": "close", "成交量": "volume", "成交额": "amount"}
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    use_cols = ["open", "high", "low", "close", "volume"]
    df = df[use_cols].astype(float)
    return df

def make_sequences_delta(df: pd.DataFrame, lookback=60,
                         x_scaler: StandardScaler = None,
                         fit_x_scaler_on_train_only=True,
                         train_end=pd.Timestamp("2025-08-29")):
    # 1. Feature standardization
    if x_scaler is None:
        x_scaler = StandardScaler()
        if fit_x_scaler_on_train_only:
            x_scaler.fit(df[df.index <= train_end])
        else:
            x_scaler.fit(df)
    df_std = pd.DataFrame(x_scaler.transform(df), 
                         index=df.index, columns=df.columns)
    
    # 2. Create sequence samples
    X_seq, X_static, Y_delta, Last_Close, Dates = [], [], [], [], []
    
    for t in range(lookback, len(df)):
        x_seq = df_std.iloc[t-lookback:t].values  # Standardized feature sequence
        x_static = np.array([1.0])  # Constant term
        
        last_close = df.iloc[t-1]["close"]  # t-1 day's closing price
        curr_close = df.iloc[t]["close"]    # t day's closing price (target)
        y_delta = curr_close - last_close   # Price residual (target)
        
        X_seq.append(x_seq)
        X_static.append(x_static)
        Y_delta.append(y_delta)
        Last_Close.append(last_close)
        Dates.append(df.index[t])
    
    # 3. Convert to arrays
    X_seq = np.array(X_seq, dtype=np.float32)
    X_static = np.array(X_static, dtype=np.float32)
    Y_delta = np.array(Y_delta, dtype=np.float32).reshape(-1, 1)
    Last_Close = np.array(Last_Close, dtype=np.float32)
    Dates = np.array(Dates)
    
    # 4. Target standardization
    y_scaler = StandardScaler()
    if fit_x_scaler_on_train_only:
        train_idx = Dates <= train_end
        y_scaler.fit(Y_delta[train_idx])
    else:
        y_scaler.fit(Y_delta)
    Y_std = y_scaler.transform(Y_delta)
    
    return X_seq, X_static, Y_std, Last_Close, Dates, x_scaler, y_scaler

def build_tf_dataset(x_seq, x_static, y, batch_size=32, shuffle=True, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices((
        {"x_seq": x_seq, "x_static": x_static},
        y
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y))
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

# =========================
# Main process
# =========================
def main():
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(True)

    # Fetch data
    df = fetch_cn1088_daily(start="20200101", end="20250930", adjust="qfq")
    print(f"[DATA] fetched rows: {len(df)}; range: {df.index.min().date()} ~ {df.index.max().date()}")

    train_end = pd.Timestamp("2025-09-30")
    pred_start = pd.Timestamp("2025-09-01")
    pred_end   = pd.Timestamp("2025-09-30")

    # Sample generation (residual target + target standardization)
    X_seq_all, X_stat_all, Ystd_all, LASTC_all, D_all, x_scaler, y_scaler = make_sequences_delta(
        df, lookback=LOOKBACK, x_scaler=None, fit_x_scaler_on_train_only=True, train_end=train_end
    )

    # Split (by target date t)
    mask_train = D_all <= train_end
    mask_pred  = (D_all >= pred_start) & (D_all <= pred_end)

    X_seq_tr, X_stat_tr, Ystd_tr = X_seq_all[mask_train], X_stat_all[mask_train], Ystd_all[mask_train]
    X_seq_pred, X_stat_pred = X_seq_all[mask_pred], X_stat_all[mask_pred]
    LASTC_pred, D_pred = LASTC_all[mask_pred], D_all[mask_pred]

    print(f"[SPLIT] train samples: {len(Ystd_tr)}; predict window samples: {len(D_pred)}")
    
    # Check prediction sample count
    if len(D_pred) == 0:
        print("[ERROR] No prediction samples found. Check date range and data availability.")
        print(f"[DEBUG] Available dates range: {D_all.min()} to {D_all.max()}")
        print(f"[DEBUG] Requested prediction range: {pred_start} to {pred_end}")
        return
    
    if len(Ystd_tr) < 200:
        raise RuntimeError("Too few training samples, check data fetching or parameter settings.")

    # Build datasets (leave 20% for validation)
    val_cut = max(int(len(Ystd_tr) * 0.8), len(Ystd_tr) - 128)
    train_ds_fit = build_tf_dataset(X_seq_tr[:val_cut], X_stat_tr[:val_cut], Ystd_tr[:val_cut],
                                    batch_size=FINAL_BATCH_SIZE, shuffle=True, repeat=False)
    valid_ds_fit = build_tf_dataset(X_seq_tr[val_cut:], X_stat_tr[val_cut:], Ystd_tr[val_cut:],
                                    batch_size=FINAL_BATCH_SIZE, shuffle=False, repeat=False)

    # Load/build model
    latest_dir = find_latest_trained_model_dir(MODELS_DIR)
    model, hp, loaded_from = try_load_meta_and_model(latest_dir)
    print("[MODEL] hparams:", hp)
    print("[MODEL] loaded_from:", loaded_from or "scratch")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mse", mode="min", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mse", mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(latest_dir if latest_dir else OUTPUT_BASE, "fine_tune_log_delta.csv"))
    ]

    print("[TRAIN] training / fine-tuning ...")
    model.fit(
        train_ds_fit,
        validation_data=valid_ds_fit,
        epochs=FINAL_EPOCHS,
        verbose=1,
        callbacks=callbacks
    )

    # Save weights
    out_dir = latest_dir if latest_dir else os.path.join(OUTPUT_BASE, "models_pso", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Keras 3.x requires using .weights.h5 extension
    fine_weights = os.path.join(out_dir, "fine_tuned_delta.weights.h5")
    
    try:
        model.save_weights(fine_weights)
        print(f"[SAVE] fine-tuned weights -> {fine_weights}")
    except Exception as e:
        print(f"[ERROR] Failed to save weights: {e}")
        # Fallback: Use English path
        fallback_dir = r"C:\Users\Henry\Desktop\code_models"
        os.makedirs(fallback_dir, exist_ok=True)
        fine_weights_fallback = os.path.join(fallback_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.weights.h5")
        model.save_weights(fine_weights_fallback)
        print(f"[SAVE] fine-tuned weights (fallback) -> {fine_weights_fallback}")

    # Prediction window: Use batch prediction instead of dataset prediction
    print(f"[PREDICT] Predicting on {len(X_seq_pred)} samples...")
    pred_ystd_list = []
    
    # Batch prediction to avoid memory issues
    for i in range(0, len(X_seq_pred), FINAL_BATCH_SIZE):
        batch_x_seq = X_seq_pred[i:i+FINAL_BATCH_SIZE]
        batch_x_stat = X_stat_pred[i:i+FINAL_BATCH_SIZE]
        batch_pred = model.predict(
            {"x_seq": batch_x_seq, "x_static": batch_x_stat},
            verbose=0
        )
        pred_ystd_list.append(batch_pred)
    
    pred_ystd = np.concatenate(pred_ystd_list, axis=0).reshape(-1, 1)
    pred_delta = y_scaler.inverse_transform(pred_ystd).reshape(-1)
    pred_price = LASTC_pred + pred_delta

    # True values (for evaluation): Take corresponding close from df
    true_close = df.loc[D_pred, "close"].astype(np.float32).values

    df_out = pd.DataFrame({
        "Date": pd.to_datetime(D_pred),
        "True_Close": true_close.astype(np.float32),
        "Pred_Close": pred_price.astype(np.float32),
    }).sort_values("Date").reset_index(drop=True)

    save_csv = os.path.join(out_dir, "predictions_CN1088_20250901_20250930_delta.csv")
    df_out.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"[EVAL] predictions saved -> {save_csv}")
    print(df_out)
    
    # Calculate evaluation metrics
    mae = np.mean(np.abs(true_close - pred_price))
    mse = np.mean((true_close - pred_price) ** 2)
    rmse = np.sqrt(mse)
    # Add R^2
    ss_res = np.sum((true_close - pred_price) ** 2)
    ss_tot = np.sum((true_close - np.mean(true_close)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    print(f"\n[METRICS] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")


if __name__ == "__main__":
    main()
