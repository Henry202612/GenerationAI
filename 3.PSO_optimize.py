import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import math
import numpy as np
import tensorflow as tf

# =========================
# Paths and Basic Configuration (consistent with your original project)
# =========================
OUTPUT_BASE = "data_saving"
CANDIDATE_TFRECORD_ROOTS = [
    os.path.join(OUTPUT_BASE, "tfrecord"),
    "data_saving/tfrecord",
]

LOOKBACK   = 60
FEAT_DIM   = 5
STATIC_DIM = 1

# Official training configuration (after PSO finds optimal hyperparameters)
FINAL_BATCH_SIZE = 512
FINAL_EPOCHS = 20

# Device & Random Seed
SEED = 20251021
USE_MIXED_PRECISION = True
ENABLE_DEVICE_PLACEMENT_LOG = False

tf.keras.utils.set_random_seed(SEED)
if ENABLE_DEVICE_PLACEMENT_LOG:
    tf.debugging.set_log_device_placement(True)

# GPU Memory Growth
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[INFO] GPUs detected: {len(gpus)} ->", gpus)
    else:
        print("[WARN] No visible GPU. Training will run on CPU.")
except Exception as e:
    print(f"[WARN] GPU init issue: {e}")

if USE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled (global_policy=mixed_float16).")

# =========================
# TFRecord Parsing (keeping consistent with your existing script)
# =========================
_FEATURE_SPEC = {
    "x_seq":       tf.io.FixedLenFeature([], tf.string),
    "x_seq_L":     tf.io.FixedLenFeature([], tf.int64),
    "x_seq_F":     tf.io.FixedLenFeature([], tf.int64),
    "x_static":    tf.io.FixedLenFeature([], tf.string),
    "x_static_S":  tf.io.FixedLenFeature([], tf.int64),
    "y":           tf.io.FixedLenFeature([], tf.float32),
}

# Data sanitization and clipping (consistent with original script)
X_CLIP = 1e4
Y_CLIP = 1e6

@tf.function
def _sanitize_tensor(x, clip_val):
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    x = tf.clip_by_value(x, -clip_val, clip_val)
    return x

def parse_example_fn(serialized):
    f = tf.io.parse_single_example(serialized, _FEATURE_SPEC)

    # x_seq: bytes(fp16) -> fp32 -> [L,F] -> Fixed static shape
    x_seq = tf.io.decode_raw(f["x_seq"], tf.float16)
    x_seq = tf.cast(x_seq, tf.float32)
    L = tf.cast(f["x_seq_L"], tf.int32)
    F = tf.cast(f["x_seq_F"], tf.int32)
    x_seq = tf.reshape(x_seq, (L, F))
    x_seq.set_shape([LOOKBACK, FEAT_DIM])

    # x_static: bytes(fp16) -> fp32 -> [S] -> Fixed static shape
    x_static = tf.io.decode_raw(f["x_static"], tf.float16)
    x_static = tf.cast(x_static, tf.float32)
    S = tf.cast(f["x_static_S"], tf.int32)
    x_static = tf.reshape(x_static, (S,))
    x_static.set_shape([STATIC_DIM])

    y = tf.reshape(f["y"], [])  # Scalar

    # Sanitization and clipping
    x_seq = _sanitize_tensor(x_seq, tf.constant(X_CLIP, tf.float32))
    x_static = _sanitize_tensor(x_static, tf.constant(X_CLIP, tf.float32))
    y = _sanitize_tensor(y, tf.constant(Y_CLIP, tf.float32))

    return {"x_seq": x_seq, "x_static": x_static}, y

def _glob_many(patterns: List[str]) -> List[str]:
    out = []
    for pat in patterns:
        out.extend(tf.io.gfile.glob(pat))
    return sorted(list(set(out)))

def resolve_files_for_split(split_dir: str) -> List[str]:
    pats = [
        os.path.join(split_dir, "*.tfrecord.gz"),
        os.path.join(split_dir, "*.tfrecord"),
        os.path.join(split_dir, "**", "*.tfrecord.gz"),
        os.path.join(split_dir, "**", "*.tfrecord"),
    ]
    return _glob_many(pats)

def _dataset_from_filelist(files: List[str], parse_fn, batch_size: int, shuffle_buf: int, repeat: bool):
    gz_files = [f for f in files if f.lower().endswith(".gz")]
    pl_files = [f for f in files if not f.lower().endswith(".gz")]

    def _build(files_subset, comp):
        if not files_subset:
            return None
        ds = tf.data.TFRecordDataset(
            files_subset,
            compression_type=comp,
            num_parallel_reads=tf.data.AUTOTUNE
        )
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    ds_gz = _build(gz_files, "GZIP")
    ds_pl = _build(pl_files, None)

    if ds_gz is not None and ds_pl is not None:
        ds = ds_gz.concatenate(ds_pl)
    elif ds_gz is not None:
        ds = ds_gz
    elif ds_pl is not None:
        ds = ds_pl
    else:
        raise FileNotFoundError("empty file list for TFRecordDataset.")

    if shuffle_buf and repeat:
        ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Handle corrupt samples without failure
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options).apply(tf.data.experimental.ignore_errors())
    return ds

def find_tfrecord_root(candidates: List[str]) -> str:
    for root in candidates:
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        if os.path.isdir(train_dir) or os.path.isdir(valid_dir):
            return root
    root = candidates[0]
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "valid"), exist_ok=True)
    return root

# =========================
# Model (Parameterized Version, for PSO Search)
# =========================
class LSTMMLP_PSO(tf.keras.Model):
    def __init__(self,
                 lookback=LOOKBACK, in_dim=FEAT_DIM, static_dim=STATIC_DIM,
                 lstm_units=64, dense1=64, dense2=32, dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.in_dim = in_dim
        self.static_dim = static_dim

        self.mask = tf.keras.layers.Masking(mask_value=0.0)
        self.lstm = tf.keras.layers.LSTM(
            int(lstm_units), return_sequences=False,
            activation="tanh", recurrent_activation="sigmoid",
            use_bias=True, recurrent_dropout=0.0, unroll=False,
            dtype="float32"  # Keep LSTM kernel in fp32
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
        x_seq = inputs["x_seq"]        # [B, L, F]
        x_static = inputs["x_static"]  # [B, S]
        x_seq.set_shape([None, self.lookback, self.in_dim])
        x_static.set_shape([None, self.static_dim])

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
        out = self.out(h, training=training)
        return out

def build_model_with_hparams(hp: Dict):
    """
    hp: {"lstm_units","dense1","dense2","dropout","lr"}
    """
    model = LSTMMLP_PSO(
        lstm_units=hp["lstm_units"],
        dense1=hp["dense1"],
        dense2=hp["dense2"],
        dropout=hp["dropout"]
    )
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae")
    ]
    base_lr = float(hp["lr"])
    if USE_MIXED_PRECISION:
        # Keep smaller LR under mixed precision
        base_lr = min(base_lr, 4e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)
    return model

# =========================
# Training/Evaluation and Saving (Reuse your original script approach)
# =========================
def safe_save_model(model: tf.keras.Model, out_dir: str):
    weights_path = os.path.join(out_dir, "final.weights.h5")
    model.save_weights(weights_path)
    print(f"[SAVE] Weights saved: {weights_path}")

    keras_file = os.path.join(out_dir, "final.keras")
    try:
        tf.keras.models.save_model(model, keras_file)
        print(f"[SAVE] Full model saved as Keras file: {keras_file}")
        return keras_file
    except Exception as e:
        print(f"[SAVE] .keras save failed -> {e}")

    savedmodel_dir = os.path.join(out_dir, "final_savedmodel")
    try:
        tf.keras.models.save_model(model, savedmodel_dir, save_format="tf")
        print(f"[SAVE] Full model saved as SavedModel dir: {savedmodel_dir}")
        return savedmodel_dir
    except Exception as e:
        print(f"[SAVE][FATAL] SavedModel save failed -> {e}")
        raise

def estimate_steps_from_manifest(tf_root: str, batch_size: int, train_files: List[str], valid_files: List[str]) -> Tuple[int, int]:
    steps_per_epoch = 2000
    val_steps = 200
    manifest_path = os.path.join(tf_root, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            mani = json.load(f)
        tr = int(max(1, mani.get("train_examples", 0) // batch_size))
        va = int(max(1, mani.get("valid_examples", 0) // batch_size))
        if tr <= 1 and len(train_files) > 0:
            tr = max(tr, min(3000, (len(train_files) * 10000) // batch_size))
        if va <= 1 and (valid_files or train_files):
            vsrc = valid_files if valid_files else train_files
            va = max(va, min(400, (len(vsrc) * 10000) // batch_size))
        steps_per_epoch = min(max(tr, 500), 3000)
        val_steps = min(max(va, 100), 400)
        print(f"[INFO] Steps (from manifest/est.): train={steps_per_epoch}, valid={val_steps}")
    except Exception:
        tr_est = (len(train_files) * 10000) // batch_size
        va_est = (len(valid_files) * 10000) // batch_size if valid_files else max(1, tr_est // 5)
        steps_per_epoch = min(max(tr_est, 500), 3000)
        val_steps = min(max(va_est, 100), 400)
        print(f"[INFO] Steps (from file-count est.): train={steps_per_epoch}, valid={val_steps}")
    return steps_per_epoch, val_steps

# =========================
# PSO Implementation
# =========================
class PSO:
    """
    Simple Particle Swarm Optimization (Global Best type) for continuous search.
    Supports mapping positions to discrete hyperparameters (e.g., network width).
    """
    def __init__(self, n_particles: int, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 w=0.72, c1=1.49, c2=1.49, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n = n_particles
        self.dim = dim
        self.lb, self.ub = bounds
        assert self.lb.shape == (dim,) and self.ub.shape == (dim,)
        # Initialization
        self.x = self.lb + (self.ub - self.lb) * self.rng.random((n_particles, dim))
        self.v = 0.1 * (self.ub - self.lb) * (self.rng.random((n_particles, dim)) - 0.5)
        self.pbest_x = self.x.copy()
        self.pbest_f = np.full((n_particles,), np.inf, dtype=np.float64)
        self.gbest_x = None
        self.gbest_f = np.inf
        self.w, self.c1, self.c2 = w, c1, c2

    def step(self, evaluate_fn):
        # Evaluation
        fvals = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            fvals[i] = evaluate_fn(self.x[i])
            if fvals[i] < self.pbest_f[i]:
                self.pbest_f[i], self.pbest_x[i] = fvals[i], self.x[i].copy()
            if fvals[i] < self.gbest_f:
                self.gbest_f, self.gbest_x = fvals[i], self.x[i].copy()

        # Update velocity and position
        r1 = self.rng.random((self.n, self.dim))
        r2 = self.rng.random((self.n, self.dim))
        cognitive = self.c1 * r1 * (self.pbest_x - self.x)
        social    = self.c2 * r2 * (self.gbest_x - self.x)
        self.v = self.w * self.v + cognitive + social
        self.x = self.x + self.v
        # Boundary bounce
        over_lb = self.x < self.lb
        over_ub = self.x > self.ub
        self.v[over_lb] *= -0.5
        self.v[over_ub] *= -0.5
        self.x = np.clip(self.x, self.lb, self.ub)
        return fvals.min(), fvals.mean(), fvals.max()

# Mapping particle position to hyperparameters
def position_to_hparams(pos: np.ndarray) -> Dict:
    """
    Dimension definitions:
    0: lstm_units in [32, 128]    -> Round to even numbers (step=8)
    1: dense1     in [32, 128]    -> Step=8
    2: dense2     in [16,  64]    -> Step=8
    3: dropout    in [0.0, 0.5]   -> Keep 2 decimal places
    4: lr(log10) in [-4.0, -2.7]  -> Learning rate 10^x, approx 1e-4 ~ 2e-3
    """
    def round_step(x, base=8, lo=None, hi=None):
        v = int(np.round(x / base) * base)
        if lo is not None: v = max(v, lo)
        if hi is not None: v = min(v, hi)
        return v

    lstm_units = round_step(pos[0], base=8, lo=32, hi=128)
    dense1     = round_step(pos[1], base=8, lo=32, hi=128)
    dense2     = round_step(pos[2], base=8, lo=16, hi=64)
    dropout    = float(np.round(pos[3], 2))
    lr         = float(10 ** pos[4])
    return {
        "lstm_units": lstm_units,
        "dense1": dense1,
        "dense2": dense2,
        "dropout": dropout,
        "lr": lr
    }

# =========================
# Build "PSO Evaluation" lightweight data flow
# =========================
def make_datasets_for_pso(train_files: List[str], valid_files: List[str]):
    """
    To accelerate PSO, each evaluation uses smaller batches and fewer steps.
    """
    PSO_BATCH = 256
    train_ds = _dataset_from_filelist(train_files, parse_example_fn, PSO_BATCH, shuffle_buf=20000, repeat=True)
    valid_ds = _dataset_from_filelist(valid_files if valid_files else train_files, parse_example_fn, PSO_BATCH, shuffle_buf=0, repeat=True)
    # Lightweight steps (can adjust for machine performance)
    steps_per_epoch = 200
    val_steps = 60
    return train_ds, valid_ds, steps_per_epoch, val_steps

# =========================
# Main Process: PSO Search + Official Training
# =========================
def main():
    # 1) Locate TFRecord
    tf_root = find_tfrecord_root(CANDIDATE_TFRECORD_ROOTS)
    train_dir = os.path.join(tf_root, "train")
    valid_dir = os.path.join(tf_root, "valid")
    print(f"[INFO] TFRecord root: {tf_root}")
    print(f"[INFO] Train dir: {train_dir}")
    print(f"[INFO] Valid dir: {valid_dir}")

    train_files = resolve_files_for_split(train_dir)
    valid_files = resolve_files_for_split(valid_dir)
    print(f"[FOUND] train files: {len(train_files)} ; valid files: {len(valid_files)}")
    if not train_files and not valid_files:
        print("[HINT] No TFRecord shards found. Please run data preprocessing to generate TFRecord.")
        return
    if not train_files and valid_files:
        print("[WARN] No train shards found, using valid shards for PSO/training (smoke test)")
        train_files = valid_files

    # 2) PSO Lightweight Dataset
    pso_train_ds, pso_valid_ds, pso_steps, pso_val_steps = make_datasets_for_pso(train_files, valid_files)

    # 3) Define PSO Search Space
    lb = np.array([32, 32, 16, 0.00, -4.0], dtype=np.float64)
    ub = np.array([128, 128, 64, 0.50, -2.7], dtype=np.float64)

    # 4) Evaluation function: return validation MSE (smaller is better)
    def eval_position(pos: np.ndarray) -> float:
        hp = position_to_hparams(pos)
        tf.keras.backend.clear_session()
        model = build_model_with_hparams(hp)
        # Few epochs for acceleration
        hist = model.fit(
            pso_train_ds,
            validation_data=pso_valid_ds,
            epochs=3,
            steps_per_epoch=pso_steps,
            validation_steps=pso_val_steps,
            verbose=0
        )
        val_mse = float(hist.history["val_mse"][-1])
        # Release resources
        del model
        return val_mse

    # 5) Run PSO
    n_particles = 12
    iters = 10
    pso = PSO(n_particles=n_particles, dim=5, bounds=(lb, ub), seed=SEED)
    print(f"[PSO] start: particles={n_particles}, iters={iters}")
    for t in range(1, iters + 1):
        best, mean, worst = pso.step(eval_position)
        print(f"[PSO][{t:02d}/{iters}] best={best:.6g}  mean={mean:.6g}  worst={worst:.6g}")
    best_hp = position_to_hparams(pso.gbest_x)
    print("[PSO] best hyperparams:", best_hp)

    # 6) Official training with the best hyperparameters
    tf.keras.backend.clear_session()
    final_model = build_model_with_hparams(best_hp)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUTPUT_BASE, "models_pso", stamp)
    os.makedirs(out_dir, exist_ok=True)

    # Estimate steps (use full batch & data)
    final_train_ds = _dataset_from_filelist(train_files, parse_example_fn, FINAL_BATCH_SIZE, shuffle_buf=20000, repeat=True)
    final_valid_ds = _dataset_from_filelist(valid_files if valid_files else train_files, parse_example_fn, FINAL_BATCH_SIZE, shuffle_buf=0, repeat=True)
    steps_per_epoch, val_steps = estimate_steps_from_manifest(tf_root, FINAL_BATCH_SIZE, train_files, valid_files)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best.weights.h5"),
            monitor="val_mse", mode="min", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mse", mode="min", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mse", mode="min", factor=0.5, patience=1, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "train_log.csv"))
    ]

    print("[TRAIN] Formal training begins ...")
    final_model.fit(
        final_train_ds,
        validation_data=final_valid_ds,
        epochs=FINAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cbs,
        verbose=1
    )

    final_path = safe_save_model(final_model, out_dir)
    eval_res = final_model.evaluate(final_valid_ds, steps=val_steps, verbose=1, return_dict=True)
    print(f"[EVAL] {eval_res}")
    print(f"[MODEL DIR] {out_dir}")

    meta = {
        "lookback": LOOKBACK,
        "feat_dim": FEAT_DIM,
        "static_dim": STATIC_DIM,
        "seed": SEED,
        "mixed_precision": USE_MIXED_PRECISION,
        "final_batch_size": FINAL_BATCH_SIZE,
        "final_epochs": FINAL_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "tfrecord_root": tf_root,
        "best_hparams": best_hp,
        "artifacts": {
            "best_weights": os.path.join(out_dir, "best.weights.h5"),
            "final_weights": os.path.join(out_dir, "final.weights.h5"),
            "final_model": final_path
        }
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
