"""Baselines and sanity checks for TrustRUL. Needs only pandas, numpy, scikit-learn
(no TensorFlow): it reads the frozen predictions in artifacts/transparency_record.csv.

  1. Simple baselines on the same test engines and the same target (RUL capped at 125).
  2. Does the "90 percent" Monte Carlo Dropout interval really cover 90 percent?
  3. Do the engines flagged for human review have larger errors than the others?

Run from the repo root:  python src/baselines_and_checks.py
Writes reports/results.json and docs/data.js (used by docs/index.html).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trustrul_core as tc

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data", ROOT / "artifacts"
S = tc.KEEP_SENSORS


def asym_score(err):
    """C-MAPSS score; err = predicted - true. Late predictions (err > 0) cost more."""
    err = np.asarray(err, dtype=float)
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


def main():
    train = tc.add_rul(tc.load_txt(DATA / "train_FD001.txt"))
    test = tc.load_txt(DATA / "test_FD001.txt")
    true = np.minimum(pd.read_csv(DATA / "RUL_FD001.txt", header=None)[0].to_numpy(), tc.RUL_CAP)

    # same 80/20 engine split as the notebook, baselines use the 80 training engines
    tr_units, _ = train_test_split(np.sort(train.unit.unique()), test_size=0.2, random_state=tc.SEED)
    tr = train[train.unit.isin(tr_units)]
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    sc = StandardScaler().fit(tr[S])
    Xtr, Xte = sc.transform(tr[S]), sc.transform(last[S])

    rows = []
    for name, m in [("Mean predictor", DummyRegressor()),
                    ("Ridge regression (last reading)", Ridge(alpha=1.0)),
                    ("Gradient boosting (last reading)", HistGradientBoostingRegressor(random_state=tc.SEED))]:
        p = m.fit(Xtr, tr["RUL"]).predict(Xte)
        e = p - true
        rows.append({"model": name, "rmse": round(float(np.sqrt(np.mean(e ** 2))), 2), "score": round(asym_score(e), 0)})

    rec = pd.read_csv(ART / "transparency_record.csv")
    e = rec["pred_RUL"] - rec["true_RUL"]
    rows.append({"model": "GRU with MC dropout (TrustRUL)", "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
                 "score": round(asym_score(e), 0)})

    inside = (rec.true_RUL >= rec.CI90_low) & (rec.true_RUL <= rec.CI90_high)
    flag = rec.review.eq("HUMAN")
    checks = {
        "n_engines": int(len(rec)),
        "interval_coverage": round(float(inside.mean()), 3),
        "nominal_coverage": 0.90,
        "mean_abs_err_flagged": round(float(rec.abs_err[flag].mean()), 2),
        "mean_abs_err_auto": round(float(rec.abs_err[~flag].mean()), 2),
        "spearman_band_vs_abs_err": round(float(rec["band"].corr(rec["abs_err"], method="spearman")), 3),
        "n_flagged": int(flag.sum()),
        "note": "true RUL is capped at 125, as in the notebook",
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps({"baselines": rows, "checks": checks}, indent=2), encoding="utf-8")

    exp = pd.read_csv(ART / "experiments.csv")
    page = {
        "baselines": rows, "checks": checks,
        "experiments": exp[["arch", "win", "dropout", "loss", "norm", "sensors", "val_RMSE"]].to_dict("records"),
        "engines": rec[["engine", "pred_RUL", "CI90_low", "CI90_high", "true_RUL", "abs_err", "review", "top_sensors"]].to_dict("records"),
    }
    (ROOT / "docs" / "data.js").write_text("window.PAGE_DATA = " + json.dumps(page) + ";\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False)); print(checks)


if __name__ == "__main__":
    main()
