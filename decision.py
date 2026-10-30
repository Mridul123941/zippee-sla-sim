"""
decision.py
===========
Step 2 of the project: turn a breach-risk number into a "tighten the geofence?" rule,
and test that rule honestly, INCLUDING its cost (orders turned away).

Key ideas
---------
* The rule: while the store's estimated breach risk is above a threshold, shrink the
  serviceable radius (5 km -> 3 km). Orders from outside 3 km are turned away.
* Store-level risk is estimated for a *reference* order from the outer ring
  (median basket, ~4 km away), given the store's current queue and rider availability.
* Evaluation is a paired simulation on fresh days the model never saw: the same
  orders are run once with a fixed 5 km radius (baseline) and once per threshold.
* Net benefit = (breaches avoided x cost per breach) - (orders turned away x margin).
  Cost numbers are PLACEHOLDERS the user can change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator import FEATURES, StoreParams, generate_day, run_day

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
EVAL_SEED_OFFSET = 10_000  # evaluation days never overlap with training days


class RiskModel:
    """Fast numpy version of the fitted scaler + logistic regression.
    (Calling sklearn once per order would be slow inside the simulation loop.)"""

    def __init__(self, pipeline):
        scaler = pipeline.named_steps["standardscaler"]
        clf = pipeline.named_steps["logisticregression"]
        self.mean = np.asarray(scaler.mean_, dtype=float)
        self.scale = np.asarray(scaler.scale_, dtype=float)
        self.coef = np.asarray(clf.coef_[0], dtype=float)
        self.intercept = float(clf.intercept_[0])

    def predict(self, feats) -> float:
        z = float(((np.asarray(feats, dtype=float) - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_named(self, queue_per_picker, items, distance_km, riders_free, is_peak) -> float:
        return self.predict((queue_per_picker, items, distance_km, riders_free, is_peak))


@dataclass
class GeofencePolicy:
    """Shrink the radius while store-level risk > threshold (with a little hysteresis)."""
    model: RiskModel
    threshold: float
    params: StoreParams
    ref_items: int = 3
    ref_distance_km: float = 4.0     # an order from the outer ring
    release_ratio: float = 0.8       # relax once risk falls below 80% of the threshold

    def __post_init__(self):
        self._tight = False

    def reset(self):
        self._tight = False

    def decide(self, feats):
        qpp, _items, dist, riders_free, is_peak = feats
        risk = self.model.predict((qpp, self.ref_items, self.ref_distance_km, riders_free, is_peak))
        if self._tight:
            if risk < self.threshold * self.release_ratio:
                self._tight = False
        elif risk > self.threshold:
            self._tight = True
        accept = not (self._tight and dist > self.params.inner_radius_km)
        return accept, self._tight


def _summarise(frames: list[pd.DataFrame]) -> dict:
    df = pd.concat(frames, ignore_index=True)
    acc = df[df["accepted"]]
    return {
        "orders_total": int(len(df)),
        "accepted": int(len(acc)),
        "rejected": int((~df["accepted"]).sum()),
        "breaches": int(acc["breach"].sum()),
        "breach_rate": float(acc["breach"].mean()) if len(acc) else 0.0,
        "peak_breach_rate": float(acc.loc[acc["is_peak"] == 1, "breach"].mean()) if (acc["is_peak"] == 1).any() else 0.0,
        "p90_delivery_min": float(acc["total_time_min"].quantile(0.90)) if len(acc) else 0.0,
        "tight_share": float(df["tight"].mean()),
    }


def run_sweep(model: RiskModel, params: StoreParams, thresholds=DEFAULT_THRESHOLDS,
              n_days: int = 30, seed: int = 42) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Paired simulation on fresh days.
    Returns (baseline_summary, sweep_table, daily_table).
    Cost assumptions are applied later (apply_costs), so changing them is instant."""
    days = [generate_day(params, seed + EVAL_SEED_OFFSET, d) for d in range(n_days)]
    base_frames = [run_day(d, params) for d in days]
    baseline = _summarise(base_frames)
    base_daily = [(int(f["accepted"].sum()), int(f.loc[f["accepted"], "breach"].sum())) for f in base_frames]

    rows, daily_rows = [], []
    for thr in thresholds:
        pol = GeofencePolicy(model, float(thr), params)
        frames = [run_day(d, params, pol) for d in days]
        s = _summarise(frames)
        s["threshold"] = float(thr)
        rows.append(s)
        for i, f in enumerate(frames):
            acc, br = int(f["accepted"].sum()), int(f.loc[f["accepted"], "breach"].sum())
            daily_rows.append({"threshold": float(thr), "day": i,
                               "breaches_avoided": base_daily[i][1] - br,
                               "orders_turned_away": base_daily[i][0] - acc})
    sweep = pd.DataFrame(rows)
    sweep["breaches_avoided"] = baseline["breaches"] - sweep["breaches"]
    sweep["orders_turned_away"] = baseline["accepted"] - sweep["accepted"]
    sweep["n_days"] = n_days
    return baseline, sweep, pd.DataFrame(daily_rows)


def apply_costs(sweep: pd.DataFrame, p_rto: float, rto_cost: float, margin_per_order: float,
                daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """net = breaches_avoided x (p_rto x rto_cost)  -  orders_turned_away x margin_per_order
    If `daily` is given, also adds net_per_day_se (standard error across simulated days)."""
    out = sweep.copy()
    cost_per_breach = p_rto * rto_cost
    out["benefit"] = out["breaches_avoided"] * cost_per_breach
    out["cost"] = out["orders_turned_away"] * margin_per_order
    out["net_benefit"] = out["benefit"] - out["cost"]
    out["net_per_day"] = out["net_benefit"] / out["n_days"]
    if daily is not None:
        d = daily.copy()
        d["net"] = d["breaches_avoided"] * cost_per_breach - d["orders_turned_away"] * margin_per_order
        g = d.groupby("threshold")["net"]
        se = (g.std(ddof=1) / np.sqrt(g.count())).rename("net_per_day_se")
        out = out.merge(se, left_on="threshold", right_index=True, how="left")
    return out


def sensitivity_grid(sweep: pd.DataFrame, p_rto_values, margin_values, rto_cost: float) -> pd.DataFrame:
    """Best achievable net benefit per day (over thresholds) for each (p_rto, margin) pair.
    If no threshold beats doing nothing, the best is 0 (i.e. leave the radius alone)."""
    recs = []
    for m in margin_values:
        for q in p_rto_values:
            net = apply_costs(sweep, q, rto_cost, m)
            best = net.loc[net["net_per_day"].idxmax()]
            val = max(float(best["net_per_day"]), 0.0)
            recs.append({"p_rto": q, "margin": m, "best_net_per_day": val,
                         "best_threshold": float(best["threshold"]) if val > 0 else np.nan})
    return pd.DataFrame(recs)


def breakeven_p_rto(sweep: pd.DataFrame, rto_cost: float, margin_per_order: float) -> float | None:
    """Smallest share of breaches-that-become-RTO at which SOME threshold has net > 0."""
    ratios = []
    for _, r in sweep.iterrows():
        if r["breaches_avoided"] > 0:
            ratios.append((r["orders_turned_away"] * margin_per_order) / (r["breaches_avoided"] * rto_cost))
    return float(min(ratios)) if ratios else None
