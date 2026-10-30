"""
app.py  -  Streamlit dashboard for the dark-store SLA-risk case study.

Run:  streamlit run app.py

Everything shown is a SYNTHETIC simulation on placeholder assumptions.
All adjustable assumptions live in the sidebar so a reader can replace them with their own.
"""
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from decision import (DEFAULT_THRESHOLDS, RiskModel, apply_costs, breakeven_p_rto,
                      run_sweep, sensitivity_grid)
from simulator import FEATURES, StoreParams
from train_sla_model import train_and_save

ART_DIR = Path(__file__).parent / "artifacts"
NAVY, TEAL, RED, GREY = "#1F3A5F", "#2A9D8F", "#D1495B", "#8A94A6"

st.set_page_config(page_title="Dark-store SLA risk: a student case study", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model (trains a fresh one if none is saved)...")
def load_artifacts():
    path = ART_DIR / "sla_model.pkl"
    try:
        art = joblib.load(path)
    except Exception:                      # missing file or scikit-learn version mismatch
        train_and_save(ART_DIR, verbose=False)
        art = joblib.load(path)
    return art


@st.cache_data(show_spinner="Running the paired simulation...")
def cached_sweep(params_dict: dict, n_days: int, _model: RiskModel):
    return run_sweep(_model, StoreParams(**params_dict), DEFAULT_THRESHOLDS, n_days=n_days)


art = load_artifacts()
risk_model = RiskModel(art["model"])
metrics = art["metrics"]
base_params: StoreParams = art["params"]

# ---------------------------------------------------------------- header
st.title("When should a dark store shrink its delivery radius?")
st.caption("A fresher's assumptions-first case study  |  Mridul Upadhyay, B.Tech Production & Industrial Engineering, PEC")
st.info(
    "**This is a simulation, not evidence.** It uses synthetic data from one imaginary dark store and "
    "placeholder numbers. It does not use any company's data. The point is to show how I would frame the "
    "problem and how sensitive the answer is to assumptions. Change the assumptions in the sidebar."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Assumptions (all placeholders)")
    p_rto = st.slider("Share of breached orders that become an RTO", 0.0, 1.0, 0.30, 0.05,
                      help="Biggest unknown. I assume a link exists; I cannot verify how strong it is.")
    rto_cost = st.slider("Cost of one RTO (Rs)", 20, 400, 100, 10)
    margin = st.slider("Margin lost per order turned away (Rs)", 0, 150, 40, 5,
                       help="Shrinking the radius means refusing some orders. This is what each one costs.")
    st.divider()
    st.header("Decision rule")
    threshold = st.select_slider("Tighten radius when breach risk exceeds", options=list(DEFAULT_THRESHOLDS),
                                 value=0.30, format_func=lambda x: f"{x:.0%}")
    st.divider()
    st.header("Store scenario")
    surge = st.slider("Evening surge multiplier", 1.2, 3.0, float(base_params.surge_multiplier), 0.1)
    rider_shift = st.slider("Rider supply vs. normal", -4, 4, 0, 1,
                            help="Fewer riders = a more stressed store.")
    n_days = st.slider("Simulated evaluation days", 10, 60, 30, 5)
    st.caption("The model is NOT retrained when you change the scenario. This shows how a model "
               "learned on one kind of store behaves when conditions shift.")

scenario = replace(base_params, surge_multiplier=surge,
                   riders_low=max(3, base_params.riders_low + rider_shift),
                   riders_high=max(4, base_params.riders_high + rider_shift))
baseline, sweep, daily = cached_sweep(asdict(scenario), n_days, risk_model)
net = apply_costs(sweep, p_rto, rto_cost, margin, daily)
best = net.loc[net["net_per_day"].idxmax()]
chosen = net.loc[(net["threshold"] - threshold).abs().idxmin()]

tabs = st.tabs(["1. Risk check", "2. Does tightening pay?", "3. Sensitivity",
                "4. Model check", "5. Assumptions & limits"])

# ---------------------------------------------------------------- tab 1
with tabs[0]:
    st.subheader("What-if: breach risk for one incoming order")
    c1, c2 = st.columns([1, 1])
    with c1:
        queue_len = st.slider("Orders waiting to be picked", 0, 60, 24)
        pickers = st.slider("Pickers on shift", 2, 8, 4)
        riders_free = st.slider("Riders free right now", 0, 20, 0)
        items = st.slider("Items in this order", 1, 12, 3)
        distance = st.slider("Distance of this order (km)", 0.3, 5.0, 4.0, 0.1)
        is_peak = st.checkbox("Peak window (8 to 11 PM)", value=True)
    risk = risk_model.predict_named(queue_len / pickers, items, distance, riders_free, int(is_peak))
    ref_risk = risk_model.predict_named(queue_len / pickers, 3, 4.0, riders_free, int(is_peak))
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100, number={"suffix": "%"},
            title={"text": "Estimated chance this order breaches 60 min"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                   "threshold": {"line": {"color": RED, "width": 4}, "value": threshold * 100},
                   "steps": [{"range": [0, threshold * 100], "color": "#DDEFEA"},
                             {"range": [threshold * 100, 100], "color": "#F6D9DE"}]}))
        fig.update_layout(height=300, margin=dict(l=30, r=50, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if ref_risk > threshold:
            st.warning(f"Store-level risk for an outer-ring order is {ref_risk:.0%}, above your {threshold:.0%} "
                       f"threshold. The rule would **suggest** tightening to {scenario.inner_radius_km:.0f} km "
                       "for an ops person to approve.")
        else:
            st.success(f"Store-level risk for an outer-ring order is {ref_risk:.0%}, below your {threshold:.0%} "
                       "threshold. The rule would keep the full radius.")
        st.caption("The red line is your chosen threshold. The rule is a suggestion for a human, not an automatic action.")

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    st.subheader("Net benefit of tightening, by threshold")
    st.write("Net benefit = (breaches avoided x cost per breach) minus (orders turned away x margin). "
             "The band is +/- 1 standard error across simulated days. Wide bands mean the result is noisy.")
    x = [f"{t:.0%}" for t in net["threshold"]]
    y = net["net_per_day"].to_numpy()
    se = net["net_per_day_se"].fillna(0).to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(y + se) + list((y - se)[::-1]), fill="toself",
                             fillcolor="rgba(42,157,143,0.18)", line=dict(width=0), hoverinfo="skip", name="+/- 1 SE"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=TEAL, width=3), name="Net benefit"))
    fig.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig.update_layout(height=380, xaxis_title="Tighten when breach risk exceeds",
                      yaxis_title="Net benefit per store-day (Rs)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if best["net_per_day"] <= 0:
        st.error("Under these assumptions no threshold beats leaving the radius alone. "
                 "Turning orders away costs more than the breaches it prevents.")
    else:
        st.success(f"Best threshold in this run: **{best['threshold']:.0%}**, about "
                   f"Rs {best['net_per_day']:.0f} per store-day (+/- {best['net_per_day_se']:.0f}). "
                   "Treat the exact number as illustrative.")
    be = breakeven_p_rto(sweep, rto_cost, margin)
    if be is not None:
        st.write(f"**Break-even:** with these costs, tightening only pays if more than about **{be:.0%}** of "
                 "breached orders turn into RTOs. That share is exactly the number real data would need to supply.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Breach rate, fixed 5 km", f"{baseline['breach_rate']:.1%}")
    m2.metric(f"Breach rate at {chosen['threshold']:.0%}", f"{chosen['breach_rate']:.1%}",
              f"{(chosen['breach_rate'] - baseline['breach_rate']) * 100:.1f} pts", delta_color="inverse")
    m3.metric("Orders turned away", f"{chosen['orders_turned_away'] / baseline['accepted']:.1%}")
    m4.metric("p90 delivery time", f"{chosen['p90_delivery_min']:.0f} min",
              f"{chosen['p90_delivery_min'] - baseline['p90_delivery_min']:.0f} min", delta_color="inverse")
    show = net[["threshold", "breach_rate", "peak_breach_rate", "breaches_avoided", "orders_turned_away",
                "net_per_day", "net_per_day_se"]].copy()
    show.columns = ["Threshold", "Breach rate", "Peak breach rate", "Breaches avoided", "Orders turned away",
                    "Net per day (Rs)", "+/- SE"]
    st.dataframe(show.style.format({"Threshold": "{:.0%}", "Breach rate": "{:.1%}", "Peak breach rate": "{:.1%}",
                                    "Net per day (Rs)": "{:.0f}", "+/- SE": "{:.0f}"}),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("How much does the answer depend on my assumptions?")
    st.write("Each cell is the best net benefit per store-day over all thresholds. Grey (0) means "
             "'leave the radius alone'. This is the honest summary: the conclusion flips as the two unknowns move.")
    p_vals = [round(v, 2) for v in np.arange(0.1, 0.81, 0.1)]
    m_vals = [10, 20, 30, 40, 60, 80, 100]
    grid = sensitivity_grid(sweep, p_vals, m_vals, rto_cost)
    z = grid.pivot(index="margin", columns="p_rto", values="best_net_per_day").loc[m_vals, p_vals]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"{v:.0%}" for v in p_vals], y=[f"Rs {m}" for m in m_vals],
                               colorscale=[[0, "#E6E9EF"], [0.001, "#D7F0EA"], [1, "#1B7F72"]],
                               text=np.round(z.values).astype(int), texttemplate="%{text}",
                               hovertemplate="RTO share %{x}<br>Margin %{y}<br>Best net: Rs %{z:.0f} per day<extra></extra>",
                               colorbar=dict(title="Rs / day")))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    fig.update_layout(height=420, xaxis_title="Share of breached orders that become an RTO",
                      yaxis_title="Margin lost per turned-away order", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"RTO cost fixed at Rs {rto_cost} (sidebar). The two numbers I most want real data on are "
               "on the axes: how often a late order becomes an RTO, and what a turned-away order really costs.")

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    st.subheader("Is the risk model any good? (on held-out days)")
    a, b, c, d = st.columns(4)
    a.metric("AUC, logistic model", f"{metrics['auc_logistic']:.3f}")
    b.metric("AUC, queue-per-picker alone", f"{metrics['auc_queue_per_picker_only']:.3f}")
    c.metric("AUC, gradient boosting (reference)", f"{metrics['auc_gradient_boosting_reference']:.3f}")
    d.metric("Brier score (lower is better)", f"{metrics['brier_logistic']:.3f}",
             help=f"Guessing the average breach rate every time scores {metrics['brier_always_predict_base_rate']:.3f}.")
    st.warning("An AUC this high is a **red flag about synthetic data, not a win**. The simulator's queue "
               "mechanics are simple, so the model recovers them easily. Real operations have rider no-shows, "
               "batching, weather and more. Treat the model as a stand-in for the decision logic.")
    cal = metrics["calibration"]
    worst = max(cal, key=lambda r: r["predicted"] - r["observed"])
    if worst["predicted"] - worst["observed"] > 0.10:
        st.warning(f"**Calibration caveat:** in the middle of the range the model over-predicts. Orders it scores at "
                   f"about {worst['predicted']:.0%} actually breach only about {worst['observed']:.0%} of the time. "
                   "The risk scores are close to 'safe' or 'overloaded' with little in between, so treat the "
                   "thresholds in this app as score cut-offs, not literal probabilities.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="Perfect"))
    fig.add_trace(go.Scatter(x=[r["predicted"] for r in cal], y=[r["observed"] for r in cal],
                             mode="lines+markers", line=dict(color=NAVY), name="Model"))
    fig.update_layout(height=340, xaxis_title="Predicted breach probability", yaxis_title="Observed breach rate",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    coefs = metrics["standardized_coefficients"]
    fig2 = go.Figure(go.Bar(x=list(coefs.values()), y=list(coefs.keys()), orientation="h",
                            marker_color=[RED if v > 0 else TEAL for v in coefs.values()]))
    fig2.update_layout(height=260, xaxis_title="Standardized coefficient (positive = more breach risk)",
                       margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Trained on {metrics['n_train_orders']:,} orders, tested on {metrics['n_test_orders']:,} orders from "
               "different days (split by day to avoid leakage). Fixed-radius breach rate on test days: "
               f"{metrics['breach_rate_test']:.1%}.")

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Assumptions I made")
    st.markdown(
        "- One dark store, evenings 5 PM to midnight, with a surge from 8 to 11 PM.\n"
        "- Orders queue FIFO for pickers, then for riders. A breach is delivery time over 60 minutes.\n"
        "- Pick-time noise and traffic noise are random and **not visible** to the model.\n"
        "- Demand decays with distance, so most orders are close to the store.\n"
        "- Tightening the radius means orders beyond the inner radius are refused while risk is high.\n"
        "- The RTO link, RTO cost and margin per order are placeholders you can change.")
    st.subheader("What this cannot tell us")
    st.markdown(
        "- Whether any of this holds in real operations. That needs real order, pick-time and rider data.\n"
        "- How customers react to being refused, or whether they come back later.\n"
        "- Rider breakdowns, weather, order batching and multi-store effects, which I left out.\n"
        "- The real link between a late delivery and an RTO, which differs by brand and by COD vs prepaid.")
    st.subheader("Questions I would ask someone who runs dark stores")
    st.markdown(
        "1. How is the serviceable radius changed today, and who decides?\n"
        "2. Which signals exist at order time (queue length, staff on floor, rider supply)?\n"
        "3. How often does a late order turn into an RTO or a cancellation?\n"
        "4. What does a refused order really cost compared with a late one?")
