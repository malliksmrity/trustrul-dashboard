# Model card: TrustRUL (coursework prototype)

Format follows Mitchell et al. (2019), "Model Cards for Model Reporting". Written by the author, October 2026.

## Model
- **Main model:** GRU network, window of 30 cycles, 14 sensor channels, dropout 0.3, Monte Carlo dropout (50 passes) for a 90 percent range. Trained with seed 42. File: `artifacts/model.keras`.
- **Reference model (added later):** gradient boosting on the last sensor reading, with split-conformal intervals and SHAP explanations (`src/conformal_shap_mlflow.py`).
- **Output:** remaining useful life in cycles (capped at 125), a range, top sensors, a human-review flag.

## Intended use
Learning and demonstration. Not for real maintenance decisions. Not validated on real engines.

## Data
NASA C-MAPSS FD001, simulated turbofan engines, one operating condition and one fault mode. 100 training engines (80 train, 20 validation) and 100 test engines. Public research data from NASA's Prognostics Center of Excellence. No personal data.

## Results (100 test engines, RUL capped at 125)
| Model | RMSE (cycles) |
|---|---|
| Mean predictor | 41.98 |
| Ridge, last reading | 20.78 |
| Gradient boosting, last reading | 17.13 |
| GRU with MC dropout | 13.47 |

Interval coverage at a nominal 90 percent: GRU saved range 82 percent, gradient boosting with split conformal 89 percent.
The review flag did not isolate the worst predictions (mean error 9.46 flagged against 10.12 not flagged).

## Limitations and risks
Simulated data only; capped target; 100 test engines; the GRU's range is too narrow; the conformal fix is shown on the baseline, not yet on the GRU; the flag rule has not been shown to help.

## EU AI Act note (unverified, not legal advice)
The coursework notebook says a system like this would be high-risk under Article 6(1) of Regulation (EU) 2024/1689 with Articles 13 and 14 obligations. My re-reading while cleaning up: civil aviation law (Regulation (EU) 2018/1139) is listed in Annex I Section B, and for Section B products Article 2(2) applies only a few AI Act provisions directly (among them Article 6(1) and Articles 102 to 109 and 112), so the Chapter III duties of Articles 13 and 14 may not apply directly. Whether a maintenance-planning tool counts as a safety component at all is a separate question. Ask a qualified person before relying on either statement. What the project does regardless: it logs the inputs behind each prediction, gives a range and a reason (top sensors), and routes uncertain cases to a person. These are good practice, not proof of compliance.

## GDPR
No personal data is used.

## Contact
github.com/malliksmrity
