"""Score maintenance policies in MaintenanceSimEnv on the 100 real run-to-failure engines, for several failure costs.

Run from the repo root:  python simulation/evaluate_policies.py      Writes simulation/results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maintenance_env import Costs, MaintenanceSimEnv, fixed_interval, never_pull, point_threshold, run_policy, uncertainty_aware

POLICIES = {
    "Never pull (run to failure)": never_pull(),
    "Fixed interval: pull at cycle 125": fixed_interval(125),
    "Fixed interval: pull at cycle 100": fixed_interval(100),
    "Point estimate: pull when predicted life <= 10": point_threshold(10),
    "Point estimate: pull when predicted life <= 20": point_threshold(20),
    "Point estimate: pull when predicted life <= 35": point_threshold(35),
    "Range-aware: pull when the lower bound <= 0 (predicted <= 33)": uncertainty_aware(0),
    "Range-aware: pull when the lower bound <= 10 (predicted <= 43)": uncertainty_aware(10),
}
out = {"engines": 100, "failure_costs_tested": [50, 150, 400], "results": {}}
for fc in out["failure_costs_tested"]:
    env = MaintenanceSimEnv(Costs(failure_cost=float(fc)))
    out["results"][str(fc)] = {name: run_policy(env, p) for name, p in POLICIES.items()}
    out["results"][str(fc)]["Hindsight optimum (pull one cycle before failure)"] = {"failures": 0, "mean_wasted_cycles_when_pulled": 1.0, "mean_cost": 1.0, "mean_score": round(1 - 1 / fc, 3)}
# risk frontier: how low can the threshold go before engines start to fail? (point estimate vs the lower end of the range)
env0 = MaintenanceSimEnv(Costs(failure_cost=150.0))
out["frontier"] = {"thresholds": list(range(0, 41, 2)), "point_estimate": [], "lower_bound": []}
for thr in out["frontier"]["thresholds"]:
    for key, pol in (("point_estimate", point_threshold(thr)), ("lower_bound", uncertainty_aware(thr))):
        r = run_policy(env0, pol)
        out["frontier"][key].append({"threshold": thr, "failures": r["failures"], "waste": r["mean_wasted_cycles_when_pulled"], "cost_at_150": r["mean_cost"]})
(Path(__file__).resolve().parent / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("--- frontier (failure cost 150): threshold, failures, mean waste")
for a, b in zip(out["frontier"]["point_estimate"], out["frontier"]["lower_bound"]):
    print(f"thr {a['threshold']:2d}  point: fail {a['failures']:3d} waste {a['waste']}   lower-bound rule: fail {b['failures']:3d} waste {b['waste']}")
for fc, r in out["results"].items():
    print(f"--- failure cost {fc}")
    for k, v in r.items():
        print(f"{k:70s} fail {v['failures']:3d}  waste {str(v['mean_wasted_cycles_when_pulled']):>6}  cost {v['mean_cost']:6.1f}  score {v['mean_score']}")
