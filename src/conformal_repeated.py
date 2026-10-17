"""Does the conformal coverage depend on which 20 engines were used to calibrate?

src/gru_conformal.py uses one split: 80 engines train the GRU, 20 calibrate. This script repeats the same
procedure with K different random splits (new GRU each time, same settings) and reports, for the same
100 test engines, the coverage of the three interval methods with mean and spread across splits.

  A. MC dropout range as is
  B. split conformal, fixed width
  C. normalised conformal (width follows the MC spread)

Needs TensorFlow (requirements-train.txt). About 4 minutes per split on CPU.
Run from the repo root:  python src/conformal_repeated.py [K]   (default 5)
Writes reports/conformal_repeated.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import trustrul_core as tc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ALPHA = 0.10
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
tf.get_logger().setLevel("ERROR")

train = tc.add_rul(tc.load_txt(DATA / "train_FD001.txt"))
test = tc.load_txt(DATA / "test_FD001.txt")
rul = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
S, win = tc.KEEP_SENSORS, tc.BEST["win"]
yte = rul.RUL.clip(upper=tc.RUL_CAP).to_numpy("float32")
cap = lambda a: np.clip(a, 0, tc.RUL_CAP)


def quantile(scores):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))


def one_split(seed):
    keras.utils.set_random_seed(seed)
    tr_units, va_units = train_test_split(np.sort(train.unit.unique()), test_size=0.2, random_state=seed)
    scaler = StandardScaler().fit(train[train.unit.isin(tr_units)][S])

    def scaled(df):
        df = df.copy()
        df[S] = scaler.transform(df[S]).astype("float32")
        return df

    tr = scaled(train)
    Xtr, ytr, _ = tc.make_windows(tr[tr.unit.isin(tr_units)], S, win, stride=1)
    Xva, yva, _ = tc.make_windows(tr[tr.unit.isin(va_units)], S, win, stride=1)
    Xte, _, _ = tc.make_windows(scaled(test), S, win, last_only=True)
    model = tc.make_model(tc.BEST["arch"], win, len(S), tc.BEST["dropout"])
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=tc.loss_fn(tc.BEST["loss"]))
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=128, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    Pva, Pte = tc.mc_predict(model, Xva, n=50), tc.mc_predict(model, Xte, n=50)
    mva, mte = Pva.mean(0), Pte.mean(0)
    bva = np.percentile(Pva, 95, axis=0) - np.percentile(Pva, 5, axis=0)
    bte = np.percentile(Pte, 95, axis=0) - np.percentile(Pte, 5, axis=0)
    lo_a, hi_a = np.percentile(Pte, [5, 95], axis=0)
    qp = quantile(np.abs(yva - mva))
    qn = quantile(np.abs(yva - mva) / np.maximum(bva, 1e-3))
    cov = lambda lo, hi: float(np.mean((yte >= lo) & (yte <= hi)))
    wid = lambda lo, hi: float(np.mean(hi - lo))
    a, b, c = (cap(lo_a), cap(hi_a)), (cap(mte - qp), cap(mte + qp)), (cap(mte - qn * bte), cap(mte + qn * bte))
    return {"seed": seed, "test_rmse": round(tc.rmse(yte, mte), 2),
            "A_cov": cov(*a), "B_cov": cov(*b), "C_cov": cov(*c), "A_w": wid(*a), "B_w": wid(*b), "C_w": wid(*c)}


rows = []
for seed in range(K):
    r = one_split(seed)
    rows.append(r)
    print(r, flush=True)
df = pd.DataFrame(rows)
summary = {m: {"coverage_mean": round(float(df[f"{m}_cov"].mean()), 3), "coverage_sd": round(float(df[f"{m}_cov"].std(ddof=1)), 3),
               "coverage_min": round(float(df[f"{m}_cov"].min()), 3), "coverage_max": round(float(df[f"{m}_cov"].max()), 3),
               "width_mean": round(float(df[f"{m}_w"].mean()), 1)} for m in "ABC"}
out = {"splits": K, "target": 1 - ALPHA, "test_engines": 100, "per_split": rows, "summary": {"A_mc_dropout": summary["A"], "B_split_conformal": summary["B"], "C_normalised_conformal": summary["C"]},
       "test_rmse_mean": round(float(df.test_rmse.mean()), 2), "test_rmse_sd": round(float(df.test_rmse.std(ddof=1)), 2)}
(ROOT / "reports" / "conformal_repeated.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out["summary"], indent=2))
