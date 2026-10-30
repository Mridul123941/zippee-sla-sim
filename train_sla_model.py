"""
train_sla_model.py
==================
Step 1 of the project: simulate orders -> train a breach-risk model -> evaluate it honestly.

Run:   python train_sla_model.py
Outputs (in ./artifacts by default):
    synthetic_orders.csv   the simulated training data
    sla_model.pkl          fitted model + metadata
    metrics.json           test-set results, calibration table, coefficients

Honesty notes
-------------
* The data is synthetic, so a good score here says "the model recovered patterns I
  built into the simulator", NOT "this works in the real world".
* Train/test split is BY DAY, because orders within one evening share the same queue
  and a random row split would leak information between train and test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simulator import FEATURES, StoreParams, simulate_orders

DEFAULT_DAYS = 36   # about 10,000 orders with the default parameters
DEFAULT_SEED = 42


def split_by_day(df: pd.DataFrame, seed: int, test_share: float = 0.25):
    days = np.array(sorted(df["day"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(days)
    n_test = max(1, int(round(len(days) * test_share)))
    test_days = set(days[:n_test].tolist())
    is_test = df["day"].isin(test_days)
    return df[~is_test].copy(), df[is_test].copy()


def calibration_table(y_true, p_pred, bins: int = 8) -> list[dict]:
    """Do predicted risks match observed breach rates? (quantile bins)"""
    d = pd.DataFrame({"y": y_true, "p": p_pred})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    out = d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True).round(4).to_dict(orient="records")


def train_and_save(out_dir: str | Path = "artifacts", n_days: int = DEFAULT_DAYS,
                   seed: int = DEFAULT_SEED, params: StoreParams | None = None,
                   verbose: bool = True) -> dict:
    params = params or StoreParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) simulate
    df = simulate_orders(params, n_days=n_days, seed=seed)
    df["breach"] = df["breach"].astype(int)
    df.drop(columns=["accepted", "tight"]).to_csv(out / "synthetic_orders.csv", index=False)
    train, test = split_by_day(df, seed)
    X_tr, y_tr = train[FEATURES], train["breach"]
    X_te, y_te = test[FEATURES], test["breach"]

    # 2) model: plain logistic regression (no class weights, so probabilities stay calibrated)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]

    # 3) baselines / sanity checks on the SAME held-out days
    base_rate = float(y_tr.mean())
    auc_model = float(roc_auc_score(y_te, p_te))
    auc_rule = float(roc_auc_score(y_te, X_te["queue_per_picker"]))        # "queue per picker" alone
    gb = GradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)      # flexible reference model
    auc_gb = float(roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]))

    scaler, clf = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    coefs = {f: round(float(c), 4) for f, c in zip(FEATURES, clf.coef_[0])}

    metrics = {
        "n_orders": int(len(df)), "n_days": int(n_days),
        "n_train_orders": int(len(train)), "n_test_orders": int(len(test)),
        "breach_rate_train": round(base_rate, 4),
        "breach_rate_test": round(float(y_te.mean()), 4),
        "auc_logistic": round(auc_model, 4),
        "auc_queue_per_picker_only": round(auc_rule, 4),
        "auc_gradient_boosting_reference": round(auc_gb, 4),
        "brier_logistic": round(float(brier_score_loss(y_te, p_te)), 4),
        "brier_always_predict_base_rate": round(float(brier_score_loss(y_te, np.full(len(y_te), base_rate))), 4),
        "log_loss_logistic": round(float(log_loss(y_te, p_te)), 4),
        "calibration": calibration_table(y_te.values, p_te),
        "standardized_coefficients": coefs,
        "features": FEATURES,
    }
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics,
                 "sklearn_version": sklearn.__version__, "params": params}, out / "sla_model.pkl")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if verbose:
        print(f"Simulated {len(df):,} orders over {n_days} days "
              f"({len(train):,} train / {len(test):,} test, split by day)")
        print(f"Breach rate: train {base_rate:.1%} | test {y_te.mean():.1%}")
        print(f"AUC  logistic: {auc_model:.3f} | queue-per-picker alone: {auc_rule:.3f} "
              f"| gradient boosting (reference): {auc_gb:.3f}")
        print(f"Brier logistic: {metrics['brier_logistic']} vs base-rate guess: "
              f"{metrics['brier_always_predict_base_rate']}")
        print("Standardized coefficients:", coefs)
        print(f"Saved artifacts to {out.resolve()}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate orders and train the SLA-breach risk model.")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default="artifacts")
    a = ap.parse_args()
    train_and_save(a.out_dir, a.days, a.seed)
