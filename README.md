# TrustRUL: engine life prediction that shows how sure it is

**In one sentence:** a neural network estimates how many flights a jet engine has left, and for every estimate it also gives a calibrated margin of error and says whether a person should take a look.

Live page: `docs/index.html` (GitHub Pages, folder `/docs`). Live engine lookup for Hugging Face Spaces: `space/`. Model card: `reports/MODEL_CARD.md`.

This repository also contains the earlier **XGBoost engine-life project** (merged in because it uses the same data).

## 1. Problem

Airlines and engine-maintenance companies decide when to pull an engine from service. Fixed intervals waste engine life; running too long risks an unplanned removal. A model that predicts the **remaining useful life (RUL)** helps, but a single number is dangerous: the planner also needs to know **how much to trust it**. The question: can a model give a remaining-life estimate with a margin of error that is actually correct, and a reason for each estimate?

Who could use it: maintenance planners in airlines or engine-service companies (plausible, not tested; the data is simulated).

## 2. Dataset

NASA C-MAPSS turbofan simulation, subset FD001: 100 training engines run to failure (20,631 rows), 100 test engines that stop early, with the true remaining life of each test engine. One operating condition, one fault mode, 21 sensors (14 informative ones kept). Source: NASA Prognostics Center of Excellence, Saxena and Goebel (2008), public for research use (check the repository terms before reuse). Files: `data/`. The true remaining life is capped at 125 cycles, as is common, because an engine far from failure looks like a healthy one.

## 3. How I solved it

1. Cut each engine's history into windows of 30 cycles over 14 sensors; the target is the remaining life at the window end.
2. Compare 12 network set-ups (LSTM, GRU, 1D-CNN with attention; windows, dropout, loss, scaling) on 20 held-back engines and pick the best.
3. Keep dropout switched on at prediction time and repeat the prediction 50 times: the spread gives a margin of error. Gradient saliency names the top sensors. Engines whose range is in the widest quarter go to a human.
4. Check the claims instead of trusting them: compare with simple baselines, measure how often the "90 percent" range really contains the truth, and check whether the human-review flag picks out the worst predictions.
5. Fix what failed: calibrate the range with split conformal prediction.
6. Compare with the earlier XGBoost project on the same truth, and test it three ways.

## 4. Findings

- **Error:** the GRU's typical error is 13.47 cycles (C-MAPSS score 354.6) on the 100 test engines. Baselines: always the mean 41.98; ridge regression 20.78; gradient boosting 17.13; XGBoost 17.76.
- **The stated range was too narrow.** The dropout range holds the truth for 82 of 100 test engines (a retrained GRU: 78), not 90.
- **Calibration fixes it.** Split conformal prediction on 20 calibration engines gives 89 of 100 (fixed width) and 90 of 100 (width proportional to the model's own spread), at a wider range: 34 cycles on average against 28. The result is uncertain by about 3 points.
- **The review flag did not isolate bad predictions:** flagged engines had a mean error of 9.46, the others 10.12. A wider range does go with a larger error to some extent (rank correlation 0.31).
- **How you test matters** (XGBoost): 35.82 with a random row split, 38.15 with whole engines held out, 25.95 on the official test (last reading only, easier). The three numbers cannot be compared with each other.
- **Explanations:** on the gradient-boosting baseline, SHAP puts sensors 11, 9 and 4 first. `cycle` is the top feature of the XGBoost model (48 percent), and without it the official-test error rises from 25.95 to 33.57.

## 5. Models used

| Model | Role | Error on test (cycles) |
|---|---|---|
| Mean predictor | baseline | 41.98 |
| Ridge regression, last reading | baseline | 20.78 |
| Gradient boosting, last reading | baseline, conformal and SHAP demo | 17.13 |
| XGBoost, labels capped at 125 | merged earlier project | 17.76 (vs capped truth) |
| LSTM, GRU, 1D-CNN + attention | compared on validation engines | best GRU: validation 11.69 |
| **GRU with Monte Carlo dropout** | main model | **13.47** |
| GRU + split conformal | calibrated range | coverage 89 to 90 of 100 (95 % Wilson range 81 to 94 %) |

## 6. Technical documentation

- **How sure is the coverage figure?** With 100 test engines, 90 covered has a 95 % Wilson range of 82.6 to 94.5 %, so it is consistent with the 90 % target. The raw dropout range (78 of 100, range 69 to 85 %) is clearly below the target. Only 20 engines calibrate the margin, so a different calibration split could shift it by a few points; a repeated-split check is the next step.

- Windows of 30 cycles, 14 sensors, standard scaling fit on the 80 training engines, RUL capped at 125, seed 42. Best configuration: GRU, dropout 0.3, MSE loss.
- Range: 50 dropout passes, 5th to 95th percentile. Review rule: range in the widest quarter of test engines.
- Conformal (`src/gru_conformal.py`): calibration set = all windows of the 20 validation engines; fixed width uses the 90th percentile of absolute errors, normalised uses the 90th percentile of error divided by the dropout band. The guarantee assumes exchangeable calibration and test points, which is only approximate here (calibration uses all cycles of the validation engines, the test uses each engine's last cycle).
- Retrained model: not bit-identical to the saved one (13.72 against 13.47).
- Baselines: `src/baselines_and_checks.py` (needs only scikit-learn); XGBoost: `src/rul_xgboost_baselines.py` (grouped CV by engine, official test set, five seeds: RMSE 18.84 Ã‚Â± 0.39 with row and column subsampling); SHAP and MLflow: `src/conformal_shap_mlflow.py`.
- EU AI Act note in the model card: the coursework notebook called the system high-risk under Article 6(1); my later re-reading is that civil-aviation law sits in Annex I Section B, where only some AI Act articles apply directly, so this is **unverified**. Ask a qualified person.
- Limits: simulated data, one operating condition and fault mode, capped target, 100 test engines, flag not shown to help, the saved `scaler.pkl` is a pickle (load only your own).
- Reproduce: `pip install -r requirements.txt`; `python src/baselines_and_checks.py`; extras `pip install -r requirements-extra.txt` then `python src/rul_xgboost_baselines.py` and `python src/conformal_shap_mlflow.py`; with TensorFlow (`requirements-train.txt`) `python src/gru_conformal.py` (about 15 minutes on CPU) and `python src/export_artifacts.py`.

## 7. Output

- Static page with six charts, the calibration comparison and an engine lookup: `docs/index.html`.
- Per-engine transparency record: predicted RUL, range, top sensors, review flag: `artifacts/transparency_record.csv`.
- Reports: `reports/results.json`, `reports/gru_conformal.json`, `reports/rul_xgboost.json`, `reports/conformal_shap.json`, `reports/MODEL_CARD.md`.
- Live lookup app: `space/app.py` (Gradio).

## 8. Deployment

**Live (Hugging Face static Space):** https://huggingface.co/spaces/malliksmrity/maintenance-sim-env

**Simulation environment (simulation/):** MaintenanceSimEnv replays the 100 real run-to-failure engines with out-of-fold predictions and a 90 percent range, and scores maintenance policies by cost. Finding: pulling when the predicted life is 8 cycles or less gives 0 failures and about 8 cycles wasted; at 4 cycles 30 of 100 engines fail; a fixed-width range rule wastes about 32 cycles. The failure cost (default 150 cycles) is an assumption, tested at 50 and 400. Live Gradio app, Space-ready, not yet deployed. n8n workflow in utomation/n8n/ (untested in n8n).

| What | How | Status |
|---|---|---|
| Static page | GitHub Pages, folder `/docs` | needs enabling |
| Live lookup | Hugging Face Space (Gradio), folder `space/`, steps in `space/README.md` | not deployed (needs your account) |
| Experiment tracking | MLflow to a local SQLite file | works locally |
| Container | `Dockerfile` runs the baselines script | built and run (Docker 29.8, 689 MB image); output matches reports/results.json: GRU RMSE 13.47, gradient boosting 17.13, interval coverage 0.82 |
| CI | `.github/workflows/ci.yml` | written, not yet run on GitHub |
| Process | `PROCESS.md` | plan, not past experience |

## Coursework and extensions

Coursework (module M507 Methods of Prediction, Gisma): the notebook `notebooks/TrustRUL.ipynb`. Extended afterwards: baselines, coverage and flag checks, conformal calibration of the GRU, SHAP and MLflow, the merged XGBoost evaluation, model card, static page and Space app. The earlier repository `predictive-maintenance-rul` is merged into this one and can be archived. Your college has approved publishing coursework (results pending).

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering. GitHub: github.com/malliksmrity. LinkedIn: linkedin.com/in/smritymallik.
