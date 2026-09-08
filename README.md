# TrustRUL — dashboard

Interactive read-out of the TrustRUL pipeline (Module M507 assessment): a GRU trained on
NASA **C-MAPSS FD001** predicts remaining useful life (RUL) for 100 held-out turbofan
engines, and every prediction carries a **transparency record** — point RUL, a Monte Carlo
Dropout 90% interval, top contributing sensors (input-gradient saliency), and an automatic
**human-review flag** when the interval is in the widest quartile.

Headline test result: **RMSE 13.47 cycles · C-MAPSS asymmetric score 354 · 25/100 engines
routed to a human.**

> Coursework demo. Not for operational maintenance decisions — MC-Dropout intervals are
> uncalibrated and the model is trained on a single operating condition only.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501. The app reads only `artifacts/` (pandas/plotly/streamlit —
no TensorFlow needed).

## Tabs

| Tab | Shows |
|---|---|
| Transparency record | Per-engine RUL, 90% interval vs true RUL, top sensors, review flag, sensor trajectory with the model input window highlighted |
| Fleet view | All 100 engines: band vs predicted RUL, threshold line, auto-vs-flagged error comparison, full record table |
| Model comparison | The §5 12-configuration table, validation-RMSE bar chart, 1D-CNN+attention weight profile |
| Live MC-Dropout | *(needs TensorFlow)* re-sample the confidence band for the selected engine with an adjustable pass count |
| About | Data source, method summary, limitations |

## Regenerating the artifact bundle

`artifacts/` is produced by `export_artifacts.py`, which reproduces the notebook's best model
(seed 42) and freezes:

```
transparency_record.csv   test_sensor_series.csv.gz   experiments.csv
attention_profile.csv     meta.json                   model.keras   scaler.pkl
```

```bash
# from the repo that also has data/train_FD001.txt etc.
python export_artifacts.py
```

## Deploy

### Streamlit Community Cloud (recommended, free)

1. Put this folder's contents at the **root** of a public GitHub repo
   (`streamlit_app.py`, `requirements.txt`, `artifacts/`, `data/`, `.streamlit/` must all be present).
2. https://share.streamlit.io → **New app** → pick the repo/branch.
3. **Main file path:** `streamlit_app.py`.
4. Deploy. First build ~2 min.
5. *(optional)* To enable the Live MC-Dropout tab, in the app's
   **Settings → rename** `requirements-live.txt` to `requirements.txt` in the repo (or copy its
   contents over) and reboot the app. Adds ~1–2 min to the build.

### Render (alternative)

1. Same repo layout. New → **Web Service** → connect the repo.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. Instance type: Free (note: free services sleep after inactivity; first hit is slow).

## Layout

```
streamlit_app.py        dashboard (reads artifacts/ only)
trustrul_core.py        model builders / windowing / MC-Dropout / metrics (mirrors the notebook)
export_artifacts.py     offline: trains best GRU, writes artifacts/
artifacts/              precomputed bundle committed to the repo
data/                   test_FD001.txt, RUL_FD001.txt  (for the live tab; public-domain NASA data)
requirements.txt        light runtime (no TensorFlow)
requirements-live.txt   light runtime + tensorflow-cpu + scikit-learn
```
