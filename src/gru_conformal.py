"""Calibrate the GRU's 90 percent range with split conformal prediction.

The saved GRU range (Monte Carlo dropout) covers only 82 of 100 test engines. This script retrains the
same GRU as src/export_artifacts.py (same seed, data split and settings) WITHOUT touching artifacts/,
then uses the 20 validation engines as a calibration set:

  A. MC dropout range as is                         (5th to 95th percentile of 50 passes)
  B. plain split conformal: mean prediction +/- q   (q = 90 % quantile of |error| on validation)
  C. normalised conformal: mean +/- q * band        (band = MC spread; q from |error| / band)

Method C keeps the width adaptive: wider when the model is unsure. It also tests whether the MC
spread is informative (if it is not, C is no better than B).

Needs TensorFlow (requirements-train.txt). About 4 minutes on CPU. Retraining is not bit-identical to
the saved model, so the RMSE here can differ slightly from the saved 13.47.

Run from the repo root:  python src/gru_conformal.py
Writes reports/gru_conformal.json and docs/gru_conformal.js.
"""
import json
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
tf.get_logger().setLevel("ERROR")
keras.utils.set_random_seed(tc.SEED)

train = tc.add_rul(tc.load_txt(DATA / "train_FD001.txt"))
test = tc.load_txt(DATA / "test_FD001.txt")
rul = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])
S, win = tc.KEEP_SENSORS, tc.BEST["win"]
tr_units, va_units = train_test_split(np.sort(train.unit.unique()), test_size=0.2, random_state=tc.SEED)
scaler = StandardScaler().fit(train[train.unit.isin(tr_units)][S])


def scaled(df):
    df = df.copy()
    df[S] = scaler.transform(df[S]).astype("float32")
    return df


tr = scaled(train)
Xtr, ytr, _ = tc.make_windows(tr[tr.unit.isin(tr_units)], S, win, stride=1)
Xva, yva, _ = tc.make_windows(tr[tr.unit.isin(va_units)], S, win, stride=1)
Xte, _, te_units = tc.make_windows(scaled(test), S, win, last_only=True)
yte = rul.RUL.clip(upper=tc.RUL_CAP).to_numpy("float32")

model = tc.make_model(tc.BEST["arch"], win, len(S), tc.BEST["dropout"])
model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=tc.loss_fn(tc.BEST["loss"]))
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=128, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

Pva = tc.mc_predict(model, Xva, n=50)
Pte = tc.mc_predict(model, Xte, n=50)
mva, mte = Pva.mean(0), Pte.mean(0)
bva = np.percentile(Pva, 95, axis=0) - np.percentile(Pva, 5, axis=0)
bte = np.percentile(Pte, 95, axis=0) - np.percentile(Pte, 5, axis=0)
lo_a, hi_a = np.percentile(Pte, [5, 95], axis=0)


def quantile(scores):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))


q_plain = quantile(np.abs(yva - mva))
q_norm = quantile(np.abs(yva - mva) / np.maximum(bva, 1e-3))
cap = lambda a: np.clip(a, 0, tc.RUL_CAP)
variants = {"A. MC dropout range as is": (cap(lo_a), cap(hi_a)),
            "B. split conformal (fixed width)": (cap(mte - q_plain), cap(mte + q_plain)),
            "C. normalised conformal (width follows MC spread)": (cap(mte - q_norm * bte), cap(mte + q_norm * bte))}
results = []
for name, (lo, hi) in variants.items():
    results.append({"method": name, "coverage": round(float(np.mean((yte >= lo) & (yte <= hi))), 3),
                    "mean_width": round(float(np.mean(hi - lo)), 1)})
# coverage on the calibration windows themselves, for method A (shows how far off MC dropout is on held-out engines)
cov_val_a = float(np.mean((yva >= np.percentile(Pva, 5, axis=0)) & (yva <= np.percentile(Pva, 95, axis=0))))
rho = float(pd.Series(bte).corr(pd.Series(np.abs(yte - mte)), method="spearman"))
out = {"retrained_gru": {"test_rmse": round(tc.rmse(yte, mte), 2), "test_score": round(float(tc.cmapss_score(yte, mte)), 1),
                         "saved_model_rmse_for_reference": 13.47},
       "calibration": {"engines": int(len(va_units)), "windows": int(len(yva)), "alpha": ALPHA, "q_fixed": round(q_plain, 1), "q_normalised": round(q_norm, 2)},
       "test_intervals": results, "mc_range_coverage_on_validation_windows": round(cov_val_a, 3),
       "spearman_mc_band_vs_abs_error_test": round(rho, 2),
       "caveats": ["calibration windows are all cycles of the validation engines; test points are the last cycle of truncated engines, so exchangeability is approximate",
                   "100 test engines: coverage has a standard error of about 3 points",
                   "retrained model is not bit-identical to the saved one"]}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "gru_conformal.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
page = dict(out, engines=[{"engine": int(u), "pred": round(float(m), 1), "true": float(t), "a_lo": round(float(a), 1), "a_hi": round(float(b), 1),
                           "c_lo": round(float(c), 1), "c_hi": round(float(d), 1)}
                          for u, m, t, a, b, c, d in zip(te_units, mte, yte, variants["A. MC dropout range as is"][0], variants["A. MC dropout range as is"][1],
                                                         variants["C. normalised conformal (width follows MC spread)"][0], variants["C. normalised conformal (width follows MC spread)"][1])])
(ROOT / "docs" / "gru_conformal.js").write_text("window.GRUC = " + json.dumps(page) + ";\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "caveats"}, indent=2))
