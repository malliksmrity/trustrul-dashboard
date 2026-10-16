"""MaintenanceSimEnv: live simulation. Replay a real C-MAPSS engine, choose when to pull it, see the cost."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from maintenance_env import Costs, MaintenanceSimEnv, fixed_interval, point_threshold, uncertainty_aware, run_policy

RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
ENV = MaintenanceSimEnv()


def plot_engine(engine, pull_cycle=None, title=""):
    d = ENV.t[ENV.t.engine == engine].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(d.cycle, d.lo, d.hi, color="#0b5d6b", alpha=0.18, label="predicted range (90 %)")
    ax.plot(d.cycle, d.pred, color="#0b5d6b", label="predicted remaining life")
    ax.plot(d.cycle, d.true_rul.clip(upper=125), color="black", linestyle="--", linewidth=1, label="true remaining life (hidden from the agent)")
    if pull_cycle is not None:
        ax.axvline(pull_cycle, color="#b35c1e", label=f"pulled at cycle {pull_cycle}")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Remaining cycles (capped at 125)"); ax.set_title(title, fontsize=10); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def manual(engine, pull_cycle, failure_cost):
    ENV.costs = Costs(failure_cost=float(failure_cost))
    r = ENV.submit(int(engine), int(pull_cycle))
    life = int(ENV.t[ENV.t.engine == int(engine)].cycle.max())
    txt = (f"Engine {int(engine)} (life {life} cycles, hidden while you decide). You pulled at cycle {int(pull_cycle)}: {r['outcome']}. "
           f"Cost {r['cost']:.0f}, score {r['score']:.2f} (1.00 = pulled one cycle before failure).")
    return txt, plot_engine(int(engine), int(pull_cycle) if r["outcome"] == "pulled" else None, "Your plan")


def auto(engine, policy, threshold, failure_cost):
    ENV.costs = Costs(failure_cost=float(failure_cost))
    pol = {"Point estimate": point_threshold(threshold), "Lower end of the range": uncertainty_aware(threshold), "Fixed interval": fixed_interval(int(threshold) * 4)}[policy]
    o, done, info = ENV.reset(int(engine)), False, {}
    while not done:
        o, r, done, info = ENV.step(pol(o))
    txt = f"Policy '{policy}' with value {threshold}: {info['outcome']} at cycle {o['cycle']}, cost {ENV.cost:.0f}."
    return txt, plot_engine(int(engine), o["cycle"] if info["outcome"] == "pulled" else None, policy)


def table(fc):
    rows = [[k, v["failures"], v["mean_wasted_cycles_when_pulled"], v["mean_cost"], v["mean_score"]] for k, v in RES["results"][str(fc)].items()]
    return rows


def frontier():
    f = RES["frontier"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(f["thresholds"], [r["failures"] for r in f["point_estimate"]], color="#0b5d6b", label="point estimate rule")
    ax.plot(f["thresholds"], [r["failures"] for r in f["lower_bound"]], color="#b35c1e", label="lower end of range rule")
    ax.set_xlabel("Threshold (cycles)"); ax.set_ylabel("Engines that fail (of 100)"); ax.legend(fontsize=8); fig.tight_layout()
    return fig


with gr.Blocks(title="MaintenanceSimEnv") as demo:
    gr.Markdown("# MaintenanceSimEnv: when should an engine be pulled?\nReal NASA C-MAPSS run-to-failure engines (simulated engines, real recorded sensor histories). "
                "Each prediction comes from a model that never saw that engine. Pulling early wastes life, running to failure costs the failure cost (an **assumption**, default 150 cycles). "
                "Code: github.com/malliksmrity/trustrul-dashboard")
    fc = gr.Slider(50, 400, value=150, step=10, label="Failure cost (cycles of life)")
    with gr.Tab("Play it yourself"):
        e1 = gr.Slider(1, 100, value=1, step=1, label="Engine")
        p1 = gr.Slider(30, 362, value=120, step=1, label="Pull at cycle")
        b1 = gr.Button("Submit plan")
        o1, f1 = gr.Textbox(label="Result"), gr.Plot()
        b1.click(manual, [e1, p1, fc], [o1, f1])
    with gr.Tab("Run a policy"):
        e2 = gr.Slider(1, 100, value=1, step=1, label="Engine")
        pol = gr.Dropdown(["Point estimate", "Lower end of the range", "Fixed interval"], value="Point estimate", label="Policy")
        th = gr.Slider(0, 40, value=10, step=1, label="Threshold in cycles (fixed interval: x4)")
        b2 = gr.Button("Run")
        o2, f2 = gr.Textbox(label="Result"), gr.Plot()
        b2.click(auto, [e2, pol, th, fc], [o2, f2])
    with gr.Tab("All 100 engines"):
        sel = gr.Radio([50, 150, 400], value=150, label="Failure cost")
        t = gr.Dataframe(headers=["Policy", "Engines that fail", "Mean wasted cycles", "Mean cost", "Mean score"], value=table(150))
        sel.change(table, sel, t)
        gr.Plot(value=frontier())
if __name__ == "__main__":
    demo.launch()
