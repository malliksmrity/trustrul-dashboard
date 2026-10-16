---
title: MaintenanceSimEnv
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# MaintenanceSimEnv: when should an engine be pulled from service?

A simulation environment built from the 100 real run-to-failure engines of NASA C-MAPSS FD001 (simulated engines with recorded sensor histories).

- **Observation:** cycle, predicted remaining life and a 90 percent range, made by a model that never saw that engine (5-fold cross-fitting).
- **Actions:** continue, or pull the engine now (`step`), or submit a pull cycle (`submit`, tool style).
- **Cost:** pulling early wastes the remaining life; running to failure costs a failure cost (an assumption, default 150 cycles, tested at 50 and 400).
- **Grade:** score = 1 - cost / failure cost. The hindsight optimum pulls one cycle before failure.

Files: `maintenance_env.py` (environment and policies), `build_predictions.py` (replay table), `evaluate_policies.py` (scores, `results.json`), `app.py` (live app).

## Deploy (you do this, with your own Hugging Face account)

1. Create a new Space: SDK Gradio, free CPU.
2. Clone it and copy everything from this `simulation/` folder into it (keep this README at the top).
3. `git add . && git commit -m "first version" && git push`.

Run locally: `pip install -r requirements.txt` then `python app.py`.