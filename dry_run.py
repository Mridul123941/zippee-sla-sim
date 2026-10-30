"""
dry_run.py  -  one-command smoke test for the whole project.

Run:  python dry_run.py
It trains a small model in a temporary folder, runs the policy sweep, and checks that the
maths is internally consistent. If this passes, the pieces fit together.
"""
import py_compile
import tempfile
from pathlib import Path

import joblib
import numpy as np

from decision import (GeofencePolicy, RiskModel, apply_costs, breakeven_p_rto, run_sweep,
                      sensitivity_grid)
from simulator import FEATURES, StoreParams, generate_day, run_day, simulate_orders
from train_sla_model import train_and_save

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


p = StoreParams()

# 1. simulator sanity -------------------------------------------------------------
df = simulate_orders(p, n_days=10, seed=1)
check("simulator produces orders", len(df) > 1000, f"{len(df)} orders")
check("breach rate is plausible (1% to 40%)", 0.01 < df["breach"].mean() < 0.40, f"{df['breach'].mean():.1%}")
check("no missing feature values", not df[FEATURES].isna().any().any())
check("same seed gives identical data", simulate_orders(p, 3, seed=7).equals(simulate_orders(p, 3, seed=7)))

# 2. training ---------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    m = train_and_save(tmp, n_days=14, seed=3, verbose=False)
    check("model beats a base-rate guess (Brier)", m["brier_logistic"] < m["brier_always_predict_base_rate"])
    check("AUC above 0.6 on held-out days", m["auc_logistic"] > 0.6, f"AUC {m['auc_logistic']:.3f}")
    check("artifacts written", all((Path(tmp) / f).exists() for f in ["sla_model.pkl", "metrics.json", "synthetic_orders.csv"]))
    art = joblib.load(Path(tmp) / "sla_model.pkl")

# 3. fast numpy risk model matches scikit-learn -----------------------------------------
pipe = art["model"]
rm = RiskModel(pipe)
sample = df[FEATURES].head(200)
sk = pipe.predict_proba(sample)[:, 1]
mine = np.array([rm.predict(r) for r in sample.to_numpy()])
check("numpy RiskModel matches sklearn", np.allclose(sk, mine, atol=1e-9), f"max diff {np.abs(sk - mine).max():.2e}")

# 4. policy is a fair, paired comparison ----------------------------------------------
day = generate_day(p, 5, 0)
base = run_day(day, p)
never = run_day(day, p, GeofencePolicy(rm, 2.0, p))          # threshold above 100%: never tightens
same = base["breach"].fillna(-1).equals(never["breach"].fillna(-1)) and never["accepted"].all()
check("a policy that never triggers reproduces the baseline exactly", same)

# 5. sweep + cost algebra -------------------------------------------------------------
baseline, sweep, daily = run_sweep(rm, p, n_days=8, seed=11)
check("baseline turns nobody away", baseline["rejected"] == 0)
check("lower thresholds turn away more orders", sweep["orders_turned_away"].iloc[0] >= sweep["orders_turned_away"].iloc[-1])
check("tightening avoids some breaches", sweep["breaches_avoided"].max() > 0)
check("per-day table adds up to the sweep",
      int(daily.groupby("threshold")["breaches_avoided"].sum().iloc[0]) == int(sweep["breaches_avoided"].iloc[0]))
net = apply_costs(sweep, 0.3, 100, 40, daily)
manual = sweep["breaches_avoided"] * 30 - sweep["orders_turned_away"] * 40
check("net benefit = avoided x (p x cost) - turned away x margin", np.allclose(net["net_benefit"], manual))
check("standard errors computed", net["net_per_day_se"].notna().all())

# 6. break-even logic is consistent -----------------------------------------------------
be = breakeven_p_rto(sweep, 100, 40)
if be is not None and be < 1.0:
    above = apply_costs(sweep, min(1.0, be * 1.05), 100, 40)["net_benefit"].max() > 0
    below = apply_costs(sweep, be * 0.95, 100, 40)["net_benefit"].max() <= 1e-9
    check("break-even flips the sign of the best net benefit", above and below, f"break-even {be:.0%}")
else:
    check("break-even defined", be is not None)
grid = sensitivity_grid(sweep, [0.2, 0.5], [20, 60], 100)
check("sensitivity grid has 4 cells and no negatives", len(grid) == 4 and (grid["best_net_per_day"] >= 0).all())

# 7. the app at least compiles (and runs, if streamlit is installed) -------------------------
py_compile.compile("app.py", doraise=True)
check("app.py compiles", True)
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120).run()
    check("app runs without exceptions (Streamlit AppTest)", not at.exception, str(at.exception) if at.exception else "")
except ImportError:
    print("[SKIP] Streamlit not installed here; run `streamlit run app.py` to see the dashboard.")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
