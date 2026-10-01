"""
EmergeRoute — Dashboard

Minimal Streamlit app tying the whole pipeline together: shows the ranked
candidate policies, the recommended top pick, and a plain-language
explanation -- matching the brief's dashboard requirement exactly.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from policy_generator import find_busiest_intersections, generate_candidate_policies
from scoring_engine import rank_policies
from traffic_prediction import train_predictor, predict_congestion_level, N_LAGS

st.set_page_config(page_title="EmergeRoute", page_icon="🚦", layout="wide")

st.title("🚦 EmergeRoute")
st.caption("AI-based traffic policy generation — proposes, simulates, and ranks traffic-control actions before recommending one.")

with st.sidebar:
    st.header("Simulation Settings")
    sumocfg = st.selectbox(
        "Traffic scenario",
        options=["simulation.sumocfg", "simulation_heavy_stable.sumocfg", "simulation_realistic.sumocfg", "simulation_heavy.sumocfg"],
        format_func=lambda x: {"simulation.sumocfg": "Light traffic", "simulation_heavy_stable.sumocfg": "Heavy congestion", "simulation_realistic.sumocfg": "Realistic (varying) traffic", "simulation_heavy.sumocfg": "Gridlock (worst-case)"}[x],
    )

    use_prediction = st.checkbox(
        "🔮 Use AI prediction (XGBoost) to set congestion level",
        value=True,
        help="Predicts near-future congestion from recent traffic history, instead of you setting it manually — this is what makes the system proactive rather than reactive.",
    )

    if not use_prediction:
        congestion_level = st.select_slider(
            "Current congestion level (drives which candidate policies get proposed)",
            options=["low", "moderate", "high"],
            value="high",
        )
    else:
        congestion_level = None  # determined at run time by the predictor

    run_button = st.button("🔍 Generate & Test Policies", type="primary", use_container_width=True)

if run_button:
    if use_prediction:
        traffic_log_map = {
            "simulation.sumocfg": "traffic_log.csv",
            "simulation_heavy.sumocfg": "traffic_log_long.csv",
            "simulation_realistic.sumocfg": "traffic_log_realistic.csv", "simulation_heavy_stable.sumocfg": "traffic_log_heavy.csv",
        }
        log_path = traffic_log_map.get(sumocfg, "traffic_log_realistic.csv")
        with st.spinner("Training prediction model on recent traffic history..."):
            model, feature_cols = train_predictor(log_path)
            df = pd.read_csv(log_path).sort_values("time_s").reset_index(drop=True)
            recent = df.tail(N_LAGS + 1)
            congestion_level = predict_congestion_level(model, feature_cols, recent)
        st.success(f"🔮 Predicted near-future congestion level: **{congestion_level.upper()}** (based on recent traffic trend, via XGBoost)")

    with st.spinner("Probing real congestion levels across the network..."):
        busiest_tls = find_busiest_intersections(sumocfg=sumocfg)

    st.success(f"Identified {len(busiest_tls)} traffic-light-controlled intersections. Busiest (by measured queue length): {', '.join(busiest_tls[:3])}")

    candidates = generate_candidate_policies(busiest_tls, congestion_level=congestion_level)

    with st.spinner(f"Testing {len(candidates)} candidate policies in SUMO simulation — this runs each one as a full simulation, may take a minute..."):
        ranked = rank_policies(candidates, sumocfg=sumocfg)

    st.header("📋 Recommended Policy")
    top = ranked[0]
    st.markdown(f"### {top['policy_name']}")
    st.info(top["explanation"])

    m = top["metrics"]
    cols = st.columns(6)
    cols[0].metric("Congestion delay", f"{m['avg_time_loss_s']:.0f}s")
    cols[1].metric("Avg travel time", f"{m['avg_travel_time_s']:.0f}s")
    cols[2].metric("Worst-case wait", f"{m['max_waiting_time_s']:.0f}s")
    cols[3].metric("CO2 emitted", f"{m['co2_kg']:.1f}kg")
    cols[4].metric("Fairness gap", f"{m['fairness_gap_s']:.0f}s")
    cols[5].metric("Trips completed", m["completed_trips"])

    st.header("📊 All Candidate Policies (ranked)")
    for entry in ranked:
        pareto_badge = " 🏆 Pareto-optimal" if entry["pareto_optimal"] else ""
        with st.expander(f"#{entry['rank']}: {entry['policy_name']}{pareto_badge}"):
            st.write(entry["explanation"])
            m = entry["metrics"]
            c = st.columns(6)
            c[0].metric("Congestion delay", f"{m['avg_time_loss_s']:.0f}s")
            c[1].metric("Avg travel time", f"{m['avg_travel_time_s']:.0f}s")
            c[2].metric("Worst-case wait", f"{m['max_waiting_time_s']:.0f}s")
            c[3].metric("CO2 emitted", f"{m['co2_kg']:.1f}kg")
            c[4].metric("Fairness gap", f"{m['fairness_gap_s']:.0f}s")
            c[5].metric("Trips completed", m["completed_trips"])

    st.caption(
        "Policies are tested via real SUMO simulation (not simulated results) and ranked "
        "using NSGA-II multi-objective optimization across congestion, travel time, safety, "
        "emergency delay, environmental impact, and fairness. 'Pareto-optimal' policies "
        "represent genuine trade-offs — no other tested policy strictly beats them on every objective."
    )
else:
    st.info("Configure a scenario in the sidebar and click **Generate & Test Policies** to run the pipeline.")
