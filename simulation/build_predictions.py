"""Build the replay table for MaintenanceSimEnv: a remaining-life prediction and a 90 percent range for EVERY cycle of
every run-to-failure engine, made by a model that has not seen that engine (5-fold cross-fitting by engine).

Model: gradient boosting on the last reading and the mean of the last 10 readings of 14 sensors; target = remaining life
capped at 125. Range: split conformal. q = 90 percent quantile of the absolute out-of-fold errors (so it is calibrated on
engines other than the one it is applied to, except that q itself pools all folds: a small simplification).

Run from the repo root:  python simulation/build_predictions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
import trustrul_core as tc

train = tc.add_rul(tc.load_txt(ROOT / "data" / "train_FD001.txt"))
S = tc.KEEP_SENSORS
life = train.groupby("unit")["cycle"].transform("max")
train["true_rul"] = life - train["cycle"]  # uncapped, used only for grading
roll = train.groupby("unit")[S].transform(lambda s: s.rolling(10, min_periods=1).mean()).add_suffix("_m10")
X = pd.concat([train[S], roll], axis=1)
y = train["RUL"]  # capped at 125
oof = np.zeros(len(train))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups=train["unit"]):
    m = HistGradientBoostingRegressor(random_state=tc.SEED).fit(X.iloc[tr], y.iloc[tr])
    oof[te] = m.predict(X.iloc[te])
q = float(np.quantile(np.abs(y.to_numpy() - oof), 0.90))
out = pd.DataFrame({"engine": train["unit"], "cycle": train["cycle"], "pred": np.clip(oof, 0, 125), "lo": np.clip(oof - q, 0, 125), "hi": np.clip(oof + q, 0, 125),
                    "true_rul": train["true_rul"]})
(ROOT / "simulation" / "data").mkdir(parents=True, exist_ok=True)
out.round(2).to_csv(ROOT / "simulation" / "data" / "engine_predictions.csv", index=False)
print(f"rows {len(out)}, engines {out.engine.nunique()}, q (90% half-width) {q:.1f}, out-of-fold RMSE {np.sqrt(np.mean((y - oof) ** 2)):.2f}")
