"""Shared TrustRUL logic — used by export_artifacts.py (offline) and, for the
optional live panel, by streamlit_app.py. Mirrors the notebook exactly.

TensorFlow is imported lazily so the dashboard runs with pandas/numpy only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants ---
COLS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

# 14 sensors retained after the notebook's near-constant / no-signal screen (§2).
KEEP_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
                "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
                "sensor_17", "sensor_20", "sensor_21"]

RUL_CAP = 125
SEED = 42
BEST = {"arch": "gru", "win": 30, "dropout": 0.3, "loss": "mse", "norm": "standard"}


# --------------------------------------------------------------- data utils ---
def load_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLS
    return df


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df = df.copy()
    eol = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (eol - df["cycle"]).clip(upper=cap).astype("float32")
    return df


def make_windows(df: pd.DataFrame, sensors, win: int, stride: int = 1, last_only: bool = False):
    """Overlapping (win, n_sensor) windows; target = RUL at the last cycle.
    Engines shorter than `win` are left-padded with a copy of their first row."""
    X, y, units = [], [], []
    has_rul = "RUL" in df.columns
    for u, g in df.groupby("unit"):
        arr = g[sensors].to_numpy("float32")
        tgt = g["RUL"].to_numpy("float32") if has_rul else None
        if len(arr) < win:
            pad = np.repeat(arr[:1], win - len(arr), axis=0)
            arr = np.vstack([pad, arr])
            if tgt is not None:
                tgt = np.concatenate([np.full(win - len(tgt), tgt[0], "float32"), tgt])
        starts = [len(arr) - win] if last_only else range(0, len(arr) - win + 1, stride)
        for s in starts:
            X.append(arr[s:s + win]); units.append(u)
            if tgt is not None:
                y.append(tgt[s + win - 1])
    X = np.asarray(X, "float32")
    y = np.asarray(y, "float32") if y else None
    return X, y, np.asarray(units)


# ------------------------------------------------------------------ metrics ---
def rmse(y_true, y_pred) -> float:
    e = np.ravel(y_pred) - np.ravel(y_true)
    return float(np.sqrt(np.mean(e ** 2)))


def cmapss_score(y_true, y_pred) -> float:
    d = np.ravel(y_pred) - np.ravel(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ------------------------------------------------------------------- models ---
def _keras():
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras


def build_rnn(win, n_feat, kind="gru", units=64, dropout=0.3):
    tf, keras = _keras()
    L = keras.layers
    Rec = L.LSTM if kind == "lstm" else L.GRU
    inp = keras.Input((win, n_feat))
    x = Rec(units, return_sequences=True)(inp); x = L.Dropout(dropout)(x)
    x = Rec(units // 2)(x);                     x = L.Dropout(dropout)(x)
    x = L.Dense(32, activation="relu")(x)
    return keras.Model(inp, L.Dense(1)(x), name=kind)


def build_cnn_attn(win, n_feat, filters=64, dropout=0.3, return_attn=False):
    tf, keras = _keras()
    L = keras.layers
    inp = keras.Input((win, n_feat))
    x = L.Conv1D(filters, 5, padding="same", activation="relu")(inp)
    x = L.Conv1D(filters, 3, padding="same", activation="relu")(x)
    x = L.Dropout(dropout)(x)
    attn = L.Softmax(axis=1, name="attn")(L.Dense(1)(x))
    ctx = L.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, attn])
    ctx = L.Dropout(dropout)(ctx)
    ctx = L.Dense(32, activation="relu")(ctx)
    out = L.Dense(1)(ctx)
    return keras.Model(inp, [out, attn] if return_attn else out, name="cnn_attn")


def make_model(arch, win, n_feat, dropout=0.3):
    return (build_rnn(win, n_feat, kind=arch, dropout=dropout)
            if arch in ("lstm", "gru") else
            build_cnn_attn(win, n_feat, dropout=dropout))


def loss_fn(kind: str):
    tf, _ = _keras()

    def _flat(y, p):
        return tf.reshape(tf.cast(y, tf.float32), [-1]), tf.reshape(p, [-1])

    def mse(y, p):
        y, p = _flat(y, p); return tf.reduce_mean(tf.square(y - p))

    def asym(y, p):
        y, p = _flat(y, p); d = p - y
        return tf.reduce_mean(tf.where(d < 0.0, tf.exp(-d / 13.0) - 1.0, tf.exp(d / 10.0) - 1.0))

    return {"mse": mse, "asym": asym}[kind]


def mc_predict(model, X, n=50):
    """MC-Dropout: n stochastic forward passes -> stacked predictions (n, N)."""
    P = []
    for _ in range(n):
        o = model(X, training=True)
        P.append(np.ravel(o[0] if isinstance(o, (list, tuple)) else o))
    return np.stack(P)


def sensor_saliency(model, X):
    """Mean |d RUL / d x| over the window -> normalised per-sensor importance (N, n_feat)."""
    tf, _ = _keras()
    xt = tf.convert_to_tensor(X)
    with tf.GradientTape() as tp:
        tp.watch(xt)
        pred = model(xt, training=False)
    g = np.abs(tp.gradient(pred, xt).numpy()).mean(axis=1)
    return g / g.sum(axis=1, keepdims=True)
