"""TrustRUL: look up one test engine, see the GRU's range, the calibrated baseline range and the sensor history."""
import matplotlib
matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

rec = pd.read_csv("transparency_record.csv").set_index("engine")
base = pd.read_csv("baseline_conformal.csv").set_index("engine")
series = pd.read_csv("test_sensor_series.csv.gz")


def show(engine, sensor):
    e = int(engine)
    r, b = rec.loc[e], base.loc[e]
    s = series[series["engine"] == e]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(s["cycle"], s[sensor], color="#0b5d6b"); a1.set_xlabel("Cycle"); a1.set_ylabel(sensor); a1.set_title("Last cycles seen by the model")
    names = ["GRU range (nominal 90%)", "Baseline, conformal (90%)"]
    a2.hlines([1, 0], [r.CI90_low, b.gb_lo], [r.CI90_high, b.gb_hi], color="#0b5d6b", linewidth=6)
    a2.plot([r.pred_RUL, b.gb_pred], [1, 0], "o", color="black"); a2.axvline(r.true_RUL, color="#b35c1e", linestyle="--", label="true value")
    a2.set_yticks([1, 0]); a2.set_yticklabels(names, fontsize=8); a2.set_xlabel("Remaining cycles (capped at 125)"); a2.legend(fontsize=8); fig.tight_layout()
    text = (f"GRU: {r.pred_RUL:.0f} cycles left, range {r.CI90_low:.0f} to {r.CI90_high:.0f}, true {r.true_RUL:.0f}. "
            f"Main sensors: {r.top_sensors}. {'Sent to a human reviewer.' if r.review == 'HUMAN' else 'Handled automatically.'}\n"
            f"Baseline (gradient boosting with split conformal): {b.gb_pred:.0f}, range {b.gb_lo:.0f} to {b.gb_hi:.0f}.\n"
            "Over all 100 test engines the GRU's range covered 82 and the conformal baseline's 89. Simulated NASA C-MAPSS data, not for real maintenance decisions.")
    return text, fig


sensors = [c for c in series.columns if c.startswith("sensor_")]
demo = gr.Interface(show, [gr.Dropdown(sorted(rec.index.tolist()), value=1, label="Test engine"), gr.Dropdown(sensors, value="sensor_11", label="Sensor")],
                    [gr.Textbox(label="Transparency record"), gr.Plot()], title="TrustRUL: engine life with a stated range",
                    description="Code and model card: github.com/malliksmrity/trustrul-dashboard")
if __name__ == "__main__":
    demo.launch()
