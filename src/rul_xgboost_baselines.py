"""XGBoost reference models for TrustRUL (merged in from the earlier predictive-maintenance-rul project).

Reproduces the original notebook number (random row split) and compares it with
evaluations that do not leak information between train and test:

  A. random 80/20 row split          (the original notebook; optimistic, see README)
  B. 5-fold CV grouped by engine     (whole engines held out)
  C. official NASA test set          (100 engines, last observed cycle, RUL_FD001.txt)

Run from the repo root:  python src/rul_xgboost_baselines.py   (needs xgboost, see requirements-extra.txt)
Writes reports/rul_xgboost.json and docs/rul_xgboost.js.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
DROP = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]  # constant sensors, as in the notebook
RUL_CAP = 125  # common practice: early in life the true RUL is not predictable


def read(name):
    return pd.read_csv(DATA / name, sep=r"\s+", header=None, names=COLS)


def with_rul(df):
    out = df.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def model():
    return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=1)


def feats(df, with_cycle=True):
    cols = [c for c in df.columns if c not in DROP + ["unit", "RUL"]]
    if not with_cycle:
        cols.remove("cycle")
    return cols


def main():
    train = with_rul(read("train_FD001.txt"))
    test = read("test_FD001.txt")
    true_rul = pd.read_csv(DATA / "RUL_FD001.txt", header=None)[0].to_numpy()
    last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
    res = []

    def add(name, split, p_rmse, p_mae=None, note=""):
        res.append({"model": name, "evaluation": split, "rmse": round(p_rmse, 2),
                    "mae": None if p_mae is None else round(p_mae, 2), "note": note})

    # A. original notebook: random row split (same engine in train and test) ---------------
    cols = feats(train)
    X = MinMaxScaler().fit_transform(train[cols])
    Xtr, Xte, ytr, yte = train_test_split(X, train["RUL"], test_size=0.2, random_state=SEED)
    m = model().fit(Xtr, ytr)
    p = m.predict(Xte)
    add("XGBoost (original)", "A. random row split", rmse(yte, p), mean_absolute_error(yte, p),
        "rows of the same engine appear in both train and test")

    # B. grouped CV by engine -----------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    def cv(make, cols_, y=train["RUL"]):
        pred = np.zeros(len(train))
        for tr, te in gkf.split(train, y, groups=train["unit"]):
            sc = MinMaxScaler().fit(train.iloc[tr][cols_])
            mm = make().fit(sc.transform(train.iloc[tr][cols_]), y.iloc[tr])
            pred[te] = mm.predict(sc.transform(train.iloc[te][cols_]))
        return pred
    for name, make, cols_ in [("Mean predictor", DummyRegressor, cols),
                              ("Linear regression (cycle only)", LinearRegression, ["cycle"]),
                              ("Linear regression (all features)", LinearRegression, cols),
                              ("XGBoost (all features)", model, cols),
                              ("XGBoost (without cycle)", model, feats(train, False))]:
        pr = cv(make, cols_)
        add(name, "B. grouped CV by engine", rmse(train["RUL"], pr), mean_absolute_error(train["RUL"], pr))

    # C. official NASA test set (last cycle of each of the 100 test engines) ----------------
    def fit_predict(make, cols_, y):
        sc = MinMaxScaler().fit(train[cols_])
        mm = make().fit(sc.transform(train[cols_]), y)
        return mm.predict(sc.transform(last[cols_])), mm
    capped = train["RUL"].clip(upper=RUL_CAP)
    preds = {}
    for name, make, cols_, y in [("Mean predictor", DummyRegressor, cols, train["RUL"]),
                                 ("Linear regression (all features)", LinearRegression, cols, train["RUL"]),
                                 ("XGBoost (all features)", model, cols, train["RUL"]),
                                 ("XGBoost (without cycle)", model, feats(train, False), train["RUL"]),
                                 ("XGBoost (RUL capped at 125)", model, cols, capped)]:
        pr, mm = fit_predict(make, cols_, y)
        preds[name] = pr
        add(name, "C. official test set (100 engines)", rmse(true_rul, pr), mean_absolute_error(true_rul, pr))
        res[-1]["rmse_vs_capped_truth"] = round(rmse(np.minimum(true_rul, RUL_CAP), pr), 2)
        if name == "XGBoost (all features)":
            imp = pd.Series(mm.feature_importances_, index=cols).sort_values(ascending=False)

    # robustness: 5 different seeds for the best model on the official test set -------------
    best = "XGBoost (RUL capped at 125)"
    seeds = []
    for sd in [1, 2, 3, 4, 5]:
        sc = MinMaxScaler().fit(train[cols])
        mm = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=sd, n_jobs=1).fit(sc.transform(train[cols]), capped)
        seeds.append(rmse(true_rul, mm.predict(sc.transform(last[cols]))))
    robustness = {"model": best + " + subsample 0.8", "seeds": [1, 2, 3, 4, 5],
                  "rmse_per_seed": [round(s, 2) for s in seeds],
                  "rmse_mean": round(float(np.mean(seeds)), 2), "rmse_std": round(float(np.std(seeds)), 2)}

    out = {"seed": SEED, "rul_cap": RUL_CAP, "n_train_rows": int(len(train)), "n_train_engines": int(train.unit.nunique()),
           "results": res, "robustness": robustness}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "rul_xgboost.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    e1 = train[train.unit == 1]
    page = {
        "results": res,
        "scatter": {"true": true_rul.tolist(), "pred": [round(float(v), 1) for v in preds[best]], "model": best},
        "importance": [{"feature": k, "value": round(float(v), 3)} for k, v in imp.head(8).items()],
        "engine1": {"cycle": e1["cycle"].tolist(), "s11": e1["s11"].round(2).tolist(), "rul": e1["RUL"].tolist()},
        "robustness": robustness,
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "rul_xgboost.js").write_text("window.RULX = " + json.dumps(page) + ";\n", encoding="utf-8")
    print(pd.DataFrame(res).to_string(index=False))
    print(robustness)


if __name__ == "__main__":
    main()
