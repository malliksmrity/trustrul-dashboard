"""Reproduce the notebook's best model (§6) and freeze everything the dashboard
needs into artifacts/ .  Run once (CPU, ~4 min):

    python src/export_artifacts.py

Outputs
-------
artifacts/transparency_record.csv  one row per test engine
artifacts/test_sensor_series.csv.gz  raw sensor values for trajectory plots
artifacts/experiments.csv          the §5 12-config table
artifacts/attention_profile.csv    1D-CNN+attention weight vs window position
artifacts/meta.json                config, thresholds, headline test metrics
artifacts/model.keras, artifacts/scaler.pkl   for the optional live panel
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
ART = HERE.parent / "artifacts"; ART.mkdir(exist_ok=True)
DATA = HERE.parent / "data"          # repo-root data/ with the C-MAPSS txt files

import trustrul_core as tc

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel("ERROR")
keras.utils.set_random_seed(tc.SEED)

# ----------------------------------------------------------------- load data ---
train = tc.add_rul(tc.load_txt(DATA / "train_FD001.txt"))
test = tc.load_txt(DATA / "test_FD001.txt")
rul = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

# sanity: hard-coded KEEP_SENSORS still matches the near-constant screen
feat = [c for c in train.columns if c not in ("unit", "cycle", "RUL")]
nunique = train[feat].nunique()
recomputed = sorted(c for c in feat if c.startswith("sensor_") and nunique[c] > 2)
assert recomputed == sorted(tc.KEEP_SENSORS), (recomputed, tc.KEEP_SENSORS)

S = tc.KEEP_SENSORS
win = tc.BEST["win"]

# --------------------------------------------------- scaler + windows (§3/§6) ---
tr_units, va_units = train_test_split(np.sort(train.unit.unique()), test_size=0.2, random_state=tc.SEED)
scaler = StandardScaler().fit(train[train.unit.isin(tr_units)][S])


def scaled(df):
    df = df.copy(); df[S] = scaler.transform(df[S]).astype("float32"); return df


tr = scaled(train)
Xtr, ytr, _ = tc.make_windows(tr[tr.unit.isin(tr_units)], S, win, stride=1)
Xva, yva, _ = tc.make_windows(tr[tr.unit.isin(va_units)], S, win, stride=1)
Xte, _, te_units = tc.make_windows(scaled(test), S, win, last_only=True)
yte = rul.RUL.clip(upper=tc.RUL_CAP).to_numpy("float32")
assert len(Xte) == len(yte) == 100

# ------------------------------------------------------- train best GRU (§6) ---
print("training best GRU ...")
model = tc.make_model(tc.BEST["arch"], win, len(S), tc.BEST["dropout"])
model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=tc.loss_fn(tc.BEST["loss"]))
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=128, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

# --------------------------------------------------- MC-Dropout + saliency -----
P = tc.mc_predict(model, Xte, n=50)
pred = P.mean(0); lo, hi = np.percentile(P, [5, 95], axis=0)
band = hi - lo
THRESH = float(np.quantile(band, 0.75))

sal = tc.sensor_saliency(model, Xte)
Sarr = np.array(S)
top_sensors = [", ".join(Sarr[np.argsort(-r)[:3]]) for r in sal]

test_rmse = tc.rmse(yte, pred); test_score = tc.cmapss_score(yte, pred)
print(f"TEST  RMSE {test_rmse:.2f}   asymmetric score {int(test_score)}")

record = pd.DataFrame({
    "engine": te_units.astype(int),
    "pred_RUL": pred.round(2),
    "CI90_low": lo.round(2), "CI90_high": hi.round(2), "band": band.round(2),
    "true_RUL": yte,
    "abs_err": np.abs(pred - yte).round(2),
    "top_sensors": top_sensors,
    "review": np.where(band > THRESH, "HUMAN", "auto"),
})
record.to_csv(ART / "transparency_record.csv", index=False)

# ------------------------------------------------------- raw sensor series -----
series = test[["unit", "cycle"] + S].rename(columns={"unit": "engine"})
series.to_csv(ART / "test_sensor_series.csv.gz", index=False, compression="gzip")

# ----------------------------------- 1D-CNN+attention temporal profile (§6) ----
print("training cnn_attn for attention profile ...")
Xtr2, ytr2, _ = tc.make_windows(tr[tr.unit.isin(tr_units)], S, win, stride=2)
keras.utils.set_random_seed(tc.SEED)
ca = tc.build_cnn_attn(win, len(S), dropout=0.3)
ca.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=tc.loss_fn("mse"))
ca.fit(Xtr2, ytr2, epochs=20, batch_size=128, verbose=0)
attn = keras.Model(ca.input, ca.get_layer("attn").output).predict(Xte, verbose=0)[..., 0]
pd.DataFrame({"window_pos": np.arange(1, win + 1),
              "mean_attention": attn.mean(0).round(4)}).to_csv(ART / "attention_profile.csv", index=False)

# ------------------------------------------------- §5 experiment table (fixed) --
# Deterministic results from the notebook run (seed 42, stride 2, 25 epochs).
experiments = pd.DataFrame([
    ("gru", 30, 0.3, "mse", "standard", "all", 20, 11.69, 3652, 9.78),
    ("gru", 30, 0.2, "mse", "standard", "all", 20, 11.84, 4074, 7.92),
    ("gru", 30, 0.4, "mse", "standard", "all", 20, 12.57, 3890, 12.07),
    ("lstm", 30, 0.3, "mse", "standard", "all", 20, 12.94, 4363, 9.78),
    ("gru", 30, 0.3, "mse", "standard", "corr", 20, 13.03, 5062, 9.87),
    ("lstm", 20, 0.3, "mse", "standard", "all", 20, 14.68, 8340, 9.47),
    ("lstm", 30, 0.3, "asym", "standard", "all", 20, 14.94, 4680, 9.05),
    ("cnn_attn", 30, 0.3, "mse", "minmax", "all", 20, 15.63, 8484, 10.20),
    ("cnn_attn", 30, 0.3, "asym", "standard", "all", 20, 21.02, 13431, 11.10),
    ("cnn_attn", 30, 0.3, "mse", "standard", "all", 20, 22.31, 25157, 10.87),
    ("cnn_attn", 40, 0.3, "mse", "standard", "all", 20, 23.65, 29307, 11.03),
    ("cnn_attn", 40, 0.4, "asym", "standard", "all", 50, 27.09, 26502, 11.94),
], columns=["arch", "win", "dropout", "loss", "norm", "sensors", "mc",
            "val_RMSE", "val_score", "mean_unc"])
experiments.to_csv(ART / "experiments.csv", index=False)

# ----------------------------------------------------------- model + meta ------
model.save(ART / "model.keras")
with open(ART / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

json.dump({
    "best": tc.BEST, "seed": tc.SEED, "rul_cap": tc.RUL_CAP, "win": win,
    "keep_sensors": S, "review_threshold_band": round(THRESH, 2),
    "n_flagged": int((record.review == "HUMAN").sum()),
    "test_rmse": round(test_rmse, 2), "test_score": int(test_score),
    "mc_passes": 50, "ci": "5th-95th percentile (90%)",
}, open(ART / "meta.json", "w"), indent=2)

print("\nwrote:", *(p.name for p in sorted(ART.iterdir())), sep="\n  ")
