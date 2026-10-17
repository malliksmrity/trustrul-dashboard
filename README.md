# TrustRUL: remaining useful life of turbofan engines with calibrated prediction intervals

Predicts how many cycles an engine has left (remaining useful life, RUL) and attaches a prediction interval whose coverage is checked, so maintenance decisions do not rest on a single number.

Live demo: https://huggingface.co/spaces/malliksmrity/maintenance-sim-env


## 1. Problem
Unplanned engine failures are costly, and early removal wastes useful life. The task is to predict RUL from sensor history and to state how uncertain the prediction is.

## 2. Dataset
NASA C-MAPSS, subset FD001: 100 training engines run to failure and 100 test engines truncated before failure, with 14 informative sensors. Simulated data; a standard benchmark. RUL labels are capped at 125 cycles, as is usual for this dataset.

## 3. Approach
1. Split the training engines 80/20 into training and validation (no engine appears in both).
2. Build 30-cycle sensor windows; scale with statistics from the training engines only.
3. Train baselines (mean, ridge regression, gradient boosting) and neural models (LSTM, GRU, 1D-CNN with attention).
4. Estimate uncertainty with Monte Carlo dropout (50 passes).
5. Calibrate the interval with split conformal prediction on the validation engines.
6. Explain predictions with SHAP; track runs with MLflow.

## 4. Results
Test RMSE in cycles, 100 test engines:

| Model | RMSE |
|---|---|
| Mean predictor | 41.98 |
| Ridge regression | 20.78 |
| Gradient boosting | 17.13 |
| GRU with MC dropout | 13.47 |

Interval coverage on the test engines (target 90 %), mean over 3 recalibrations with different validation engines:

| Interval | Coverage | Range |
|---|---|---|
| Raw MC dropout | 79 % | 75 to 83 % |
| Split conformal | 88 % | 86 to 89 % |
| Normalised conformal | 89 % | 88 to 91 % |

The retrained models averaged an RMSE of 14.35 (13.93 to 14.61), above the saved model's 13.47, so the saved run is on the favourable side.

## 5. Technical notes
- Window 30, 14 sensors, seed 42; GRU with dropout 0.3.
- Conformal: absolute-error quantile on validation windows; the normalised variant divides by the Monte Carlo spread.
- Coverage on 100 test engines has a standard error of about 3 points (95 % Wilson interval for 90 of 100: 83 to 94 %).
- Reproduce: `pip install -r requirements.txt`, then `python src/baselines_and_checks.py`. Retraining needs `requirements-train.txt` (TensorFlow). A Dockerfile reproduces the baseline table.

## 6. Limitations
- One benchmark subset with simulated degradation; real fleets differ.
- Only 20 engines calibrate the interval, so coverage lands slightly below target.
- Conformal guarantees assume exchangeable data, which holds only approximately here.
- The interactive demo's fan-speed slider is an illustration; the dataset's fan speed barely varies.

## 7. Deployment
| Item | Status |
|---|---|
| Static page (GitHub Pages, `/docs`) | live |
| Hugging Face static Space (3D maintenance simulator) | live |
| Dockerfile | built and run; output matches `reports/results.json` |
| CI (GitHub Actions) | passing |

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering.

