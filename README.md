# TrustRUL: engine life prediction that shows how sure it is

**In one sentence:** a neural network estimates how many flights a jet engine has left, and for every estimate it also says how unsure it is and whether a person should take a look.

Project page (charts, plain language): `docs/index.html`, served with GitHub Pages once Pages is enabled for the `/docs` folder.

## The question

Can a model give maintenance planners a remaining-life estimate that comes with a margin of error and a "please check this one" flag, instead of a single unexplained number?

## Who could benefit, and how

Maintenance planners at airlines or engine service companies could, in principle, use such a tool to decide when to inspect an engine and which predictions to double-check. That is plausible but not tested: the data is simulated.

## Data and licence

- NASA C-MAPSS turbofan simulation, subset FD001 (100 training engines, 100 test engines, one operating condition, one fault type).
- Source: NASA Prognostics Center of Excellence. Cite: Saxena, A. and Goebel, K. (2008). Public for research use; check the repository terms before reuse.
- Files: `data/train_FD001.txt`, `data/test_FD001.txt`, `data/RUL_FD001.txt`.

## How it works

1. Read the last 30 cycles of 14 sensor channels (constant sensors are dropped).
2. A GRU network predicts the remaining cycles. Dropout stays on and the prediction is repeated 50 times, which gives a 5th to 95th percentile range.
3. Sensor importance comes from the gradient of the output with respect to the inputs.
4. If the range is in the widest quarter of all test engines, the engine is routed to a human.

## Results

Test set: the 100 official test engines. The true remaining life is capped at 125 cycles, as in the notebook. Computed by `python src/baselines_and_checks.py` from the saved predictions, saved in `reports/results.json`.

| Model | RMSE (cycles) | C-MAPSS score (lower is better) |
|---|---|---|
| Mean predictor | 41.98 | 33781 |
| Ridge regression, last reading | 20.78 | 1236 |
| Gradient boosting, last reading | 17.13 | 917 |
| GRU with MC dropout (TrustRUL) | **13.47** | **354.6** |

The baselines are an addition: they train on the same 80 engines as the notebook and use only the latest sensor reading. Twelve network set-ups were compared on 20 validation engines (`artifacts/experiments.csv`). The GRU with a window of 30 cycles had the lowest validation RMSE, 11.69.

**Checks of the uncertainty part (new):**

- The "90 percent" range contained the true value for 82 of 100 engines. It is a bit too narrow.
- The 25 engines flagged for human review have an average error of 9.46 cycles. The 75 others have 10.12. So the flag did not isolate the worst predictions in this test. The width of the range and the error are positively related (Spearman 0.31).

**Calibrated range, explanations, tracking (added in the second cleanup round)**

`python src/conformal_shap_mlflow.py` (needs `requirements-extra.txt`; reports in `reports/conformal_shap.json`):

- Split conformal prediction on the gradient-boosting baseline, calibrated on the 20 validation engines: the 90 percent range has half-width 29.8 cycles and covers 89 of 100 test engines (the GRU's saved range: 82). The result is uncertain by about 3 points. It applies to the baseline. Doing the same for the GRU needs retraining it first (not done).
- SHAP (TreeExplainer) on the baseline: sensors 11, 9 and 4 contribute most on average.
- The run is logged with MLflow to a local SQLite file (`mlflow.db`, git-ignored). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- A model card with an EU AI Act note is in `reports/MODEL_CARD.md`. The note says the notebook's high-risk claim is unverified.

## Limitations

- Simulated data, a single operating condition and fault type, 100 test engines only.
- True remaining life is capped at 125, so scores are not comparable with papers that use uncapped targets.
- Baselines were not tuned. The GRU configuration was picked on validation data, not the test engines.
- The network was not retrained for this cleanup (TensorFlow is large). `src/export_artifacts.py` retrains it and rewrites `artifacts/`. The numbers here are computed from the saved artifacts.
- The saved `scaler.pkl` is a Python pickle. Only load pickles you created yourself.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/baselines_and_checks.py     # baselines, coverage check, writes docs/data.js
# optional, needs TensorFlow, about 4 minutes on CPU:
pip install -r requirements-train.txt
python src/export_artifacts.py
```

## What is coursework and what I extended

- Coursework (module M507 Methods of Prediction, Gisma): the notebook `notebooks/TrustRUL.ipynb` with the model comparison, MC dropout and the transparency record.
- Extended afterwards: the first dashboard, and in this cleanup the baselines, the interval coverage check, the flag check and the static page. The Streamlit app is in `_archive/`.
- Check your university's rules before publishing coursework.

## For technical readers

- Window 30, standard scaling fit on the 80 training engines, RUL target capped at 125, seed 42.
- Best configuration: GRU, dropout 0.3, MSE loss. Test RMSE 13.47 and score 354.6 (the saved `artifacts/meta.json` shows 354, truncated).
- Transparency record per engine: predicted RUL, MC-dropout 5th and 95th percentile, top three sensors by input-gradient saliency, review flag. See `artifacts/transparency_record.csv`.
- The notebook argues that a system like this would count as high-risk under the EU AI Act. That is the author's reading for the coursework, not legal advice.

## Repository layout

```
data/        C-MAPSS FD001 files
notebooks/   TrustRUL.ipynb (executed notebook)
src/         trustrul_core.py, export_artifacts.py, baselines_and_checks.py
artifacts/   saved predictions, experiment table, model, scaler
reports/     results.json
docs/        static project page
_archive/    moved files: Streamlit app and its config, original README
```

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering.
GitHub: github.com/malliksmrity. LinkedIn: linkedin.com/in/smritymallik.

## Process and automation

- `PROCESS.md`: backlog, definition of done and sprint rhythm for the next work (a plan, not past experience).
- `.github/workflows/ci.yml`: GitHub Actions check (written here, not yet run on GitHub).
- `Dockerfile`: container for the main script (not built or tested yet).

## Live model UI (Hugging Face Space)

`space/` is a ready-to-deploy Gradio app for a Hugging Face Space. Instructions in `space/README.md`. Not deployed yet: it needs your Hugging Face account.
