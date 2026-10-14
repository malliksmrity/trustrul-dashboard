"""Calibrated intervals, explanations and experiment tracking for TrustRUL (no TensorFlow needed).

The GRU's saved intervals cover only 82 percent of true values instead of 90. This script shows
a standard fix, split conformal prediction, and applies it to the gradient-boosting baseline,
because calibrating the GRU needs its validation-set predictions, which require retraining it
(src/export_artifacts.py). It also explains the baseline with SHAP and logs the run with MLflow.

  1. Fit gradient boosting on the 80 training engines (same split as the notebook).
  2. Use the 20 validation engines as a calibration set: the 90 percent quantile of absolute errors
     becomes the half-width of every interval.
  3. Check the coverage on the 100 test engines.
  4. SHAP values for the 14 sensors on the test engines.
  5. Log parameters and metrics to a local MLflow SQLite file (mlflow.db, git-ignored).

Run from the repo root:  python src/conformal_shap_mlflow.py     (needs requirements-extra.txt)
Writes reports/conformal_shap.json and docs/conformal.js.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trustrul_core as tc

ROOT = Path(__file__).resolve().parents[1]
S, ALPHA = tc.KEEP_SENSORS, 0.10


def main():
    import mlflow
    import shap

    train = tc.add_rul(tc.load_txt(ROOT / "data" / "train_FD001.txt"))
    test = tc.load_txt(ROOT / "data" / "test_FD001.txt")
    true = np.minimum(pd.read_csv(ROOT / "data" / "RUL_FD001.txt", header=None)[0].to_numpy(), tc.RUL_CAP)
    tr_u, va_u = train_test_split(np.sort(train.unit.unique()), test_size=0.2, random_state=tc.SEED)
    tr, va = train[train.unit.isin(tr_u)], train[train.unit.isin(va_u)]
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    sc = StandardScaler().fit(tr[S])
    model = GradientBoostingRegressor(random_state=tc.SEED).fit(sc.transform(tr[S]), tr["RUL"])

    cal_err = np.abs(va["RUL"].to_numpy() - model.predict(sc.transform(va[S])))
    n = len(cal_err)
    q = float(np.quantile(cal_err, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    pred = model.predict(sc.transform(last[S]))
    lo, hi = np.clip(pred - q, 0, tc.RUL_CAP), np.clip(pred + q, 0, tc.RUL_CAP)
    coverage = float(np.mean((true >= lo) & (true <= hi)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(sc.transform(last[S]))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=S).sort_values(ascending=False)

    out = {"model": "GradientBoostingRegressor on last reading, 14 sensors", "target": "RUL capped at 125",
           "calibration_engines": int(len(va_u)), "calibration_rows": int(n), "alpha": ALPHA,
           "half_width_cycles": round(q, 1), "test_rmse": round(rmse, 2), "test_coverage": round(coverage, 3),
           "gru_saved_interval_coverage": 0.82,
           "shap_mean_abs_top": {k: round(float(v), 2) for k, v in imp.head(8).items()},
           "caveats": ["calibration rows are all cycles of the validation engines, test points are the last cycle of truncated engines, so exchangeability is only approximate",
                       "100 test engines: coverage has a standard error of about 3 points",
                       "applies to the baseline, not to the GRU"]}
    (ROOT / "reports" / "conformal_shap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    page = dict(out, engines=[{"engine": int(u), "pred": round(float(p), 1), "lo": round(float(l), 1), "hi": round(float(h), 1), "true": float(t)}
                              for u, p, l, h, t in zip(last["unit"], pred, lo, hi, true)])
    (ROOT / "docs" / "conformal.js").write_text("window.CONFORMAL = " + json.dumps(page) + ";\n", encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("trustrul-baseline-conformal")
    with mlflow.start_run(run_name="gbr_split_conformal"):
        mlflow.log_params({"model": "GradientBoostingRegressor", "seed": tc.SEED, "alpha": ALPHA, "rul_cap": tc.RUL_CAP})
        mlflow.log_metrics({"test_rmse": rmse, "test_coverage": coverage, "half_width": q})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
