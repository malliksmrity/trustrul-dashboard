"""MaintenanceSimEnv: when should an engine be pulled from service?

Built from real records: the 100 run-to-failure engines of NASA C-MAPSS FD001. At every cycle the agent sees a
remaining-life prediction with a 90 percent range, made by a model that has NOT seen that engine (5-fold
cross-fitting, see simulation/build_predictions.py), and chooses: continue, or pull the engine now.

Costs (in "cycles of engine life", the unit of the data):
  pull early:  wasted life = cycles the engine could still have run            (waste_per_cycle each)
  run to failure: failure_cost                                               (ASSUMPTION, default 150 cycles; vary it)
Reward is the negative cost. Score in [0, 1]: 1 - cost / failure_cost (1 = pulled at the last safe cycle).
The hindsight optimum is to pull one cycle before failure (cost 1 cycle of waste); no real policy knows it.

API: reset(engine) / step(action) (gym-style), plus tool-style get_situation() / submit(pull_cycle) for agents.
No third-party dependency except pandas and numpy.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "engine_predictions.csv"
CONTINUE, PULL = 0, 1


@dataclass
class Costs:
    failure_cost: float = 150.0   # assumption: one failure costs as much as 150 cycles of unused life
    waste_per_cycle: float = 1.0
    min_cycle: int = 30           # no pulls before this cycle


class MaintenanceSimEnv:
    def __init__(self, costs: Costs = None, table: pd.DataFrame = None):
        self.costs = costs or Costs()
        self.t = table if table is not None else pd.read_csv(DATA)
        self.engines = sorted(self.t.engine.unique())
        self.df = None
        self.i = 0
        self.done = True

    # --- gym-style -----------------------------------------------------------------------
    def reset(self, engine: int):
        self.df = self.t[self.t.engine == engine].sort_values("cycle").reset_index(drop=True)
        self.i, self.done, self.cost = 0, False, 0.0
        return self._obs()

    def _obs(self):
        r = self.df.iloc[self.i]
        return {"engine": int(r.engine), "cycle": int(r.cycle), "pred_rul": float(r.pred), "lo": float(r.lo), "hi": float(r.hi)}

    def step(self, action: int):
        assert not self.done, "episode finished: call reset()"
        r = self.df.iloc[self.i]
        last = self.i == len(self.df) - 1
        # an engine fails at the end of its last cycle, so pulling on that cycle is too late (same rule as submit())
        if action == PULL and r.cycle >= self.costs.min_cycle and not last:
            self.done = True
            self.cost = self.costs.waste_per_cycle * float(r.true_rul)
            return self._obs(), -self.cost, True, {"outcome": "pulled", "wasted_cycles": float(r.true_rul)}
        if last:  # the engine fails at its final cycle
            self.done = True
            self.cost = self.costs.failure_cost
            return self._obs(), -self.cost, True, {"outcome": "failed", "wasted_cycles": 0.0}
        self.i += 1
        return self._obs(), 0.0, False, {}

    # --- tool-style, for agents ----------------------------------------------------------
    def get_situation(self, engine: int):
        d = self.t[self.t.engine == engine].sort_values("cycle")
        return {"engine": engine, "life_cycles": int(d.cycle.max()), "history": d[["cycle", "pred", "lo", "hi"]].round(1).to_dict("records"),
                "costs": self.costs.__dict__, "note": "true remaining life is hidden"}

    def submit(self, engine: int, pull_cycle: int):
        """Grade one plan: pull at pull_cycle (None = never). Returns cost and score."""
        d = self.t[self.t.engine == engine].sort_values("cycle")
        life = int(d.cycle.max())
        if pull_cycle is None or pull_cycle >= life or pull_cycle < self.costs.min_cycle:
            cost = self.costs.failure_cost if (pull_cycle is None or pull_cycle >= life) else self.costs.failure_cost
            outcome = "failed" if (pull_cycle is None or pull_cycle >= life) else "invalid (before minimum cycle)"
            return {"outcome": outcome, "cost": cost, "score": 0.0, "optimal_cost": self.costs.waste_per_cycle}
        waste = self.costs.waste_per_cycle * (life - pull_cycle)
        return {"outcome": "pulled", "wasted_cycles": life - pull_cycle, "cost": waste, "score": max(0.0, 1 - waste / self.costs.failure_cost),
                "optimal_cost": self.costs.waste_per_cycle}


# --- policies (each maps an observation to an action) ---------------------------------------
def fixed_interval(n=125):
    return lambda o: PULL if o["cycle"] >= n else CONTINUE


def point_threshold(thr=20):
    return lambda o: PULL if o["pred_rul"] <= thr else CONTINUE


def uncertainty_aware(buffer=5):
    """Pull when even the optimistic end (upper bound) of the range is within `buffer` cycles of failure... i.e. when
    the lower bound says it might be close: lo <= buffer."""
    return lambda o: PULL if o["lo"] <= buffer else CONTINUE


def never_pull():
    return lambda o: CONTINUE


def run_policy(env: MaintenanceSimEnv, policy):
    res = []
    for e in env.engines:
        o, done, info = env.reset(e), False, {}
        while not done:
            o, r, done, info = env.step(policy(o))
        res.append({"engine": e, "outcome": info["outcome"], "cost": env.cost, "wasted": info["wasted_cycles"]})
    d = pd.DataFrame(res)
    return {"failures": int((d.outcome == "failed").sum()), "mean_wasted_cycles_when_pulled": round(float(d[d.outcome == "pulled"].wasted.mean()), 1) if (d.outcome == "pulled").any() else None,
            "mean_cost": round(float(d.cost.mean()), 1), "mean_score": round(float(np.mean(np.maximum(0, 1 - d.cost / env.costs.failure_cost))), 3)}
