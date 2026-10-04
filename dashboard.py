"""
EmergeRoute - Dashboard

Streamlit app tying the whole pipeline together: shows real network state,
the ranked candidate policies, the recommended top pick, and a plain-language
explanation, matching the brief's dashboard requirement.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations
from detection_panel import render_detection_section
from video_upload import render_video_upload

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from policy_generator import probe_network, generate_candidate_policies
from scoring_engine import rank_policies
from traffic_prediction import train_predictor, predict_congestion_level, N_LAGS

st.set_page_config(page_title="EmergeRoute", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Theme + scroll-reveal (pure CSS, no JS -- Streamlit strips <script> tags
# from markdown, and components.html runs in a sandboxed iframe that can't
# see the main page's scroll position, so JS-based scroll triggers genuinely
# don't work here. animation-timeline: view() is native CSS, supported in
# Chrome/Brave, and gives a real scroll-triggered reveal without any of that.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e6e9ef; }
    #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 2rem; max-width: 1280px; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

    .topbar {
        background-color: #121722;
        border: 1px solid #1f2430;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.6rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }

    .card {
        background-color: #121722;
        border: 1px solid #1f2430;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .card-label {
        font-size: 0.78rem;
        color: #8b93a7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .card-value { font-size: 1.6rem; font-weight: 600; color: #f2f4f8; }

    .stat-bar { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1.1rem; }
    .stat-pill {
        background-color: #0f1319;
        border: 1px solid #1f2430;
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        font-size: 0.85rem;
        color: #c7cdda;
    }
    .stat-pill b { color: #f2f4f8; }

    .status-banner {
        background-color: #121722;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.7rem;
        font-size: 0.95rem;
    }
    .status-banner.ok { border-left-color: #22c55e; }

    .pareto-tag {
        display: inline-block;
        background-color: #14532d;
        color: #86efac;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.12rem 0.5rem;
        border-radius: 4px;
        margin-left: 0.5rem;
        letter-spacing: 0.02em;
    }

    .zone-card {
        background-color: #121722;
        border: 1px solid #1f2430;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .zone-id { font-size: 0.95rem; font-weight: 600; color: #f2f4f8; }
    .zone-queue { font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }
    .zone-label { font-size: 0.72rem; color: #8b93a7; text-transform: uppercase; letter-spacing: 0.03em; }

    .event-row {
        border-left: 3px solid #f59e0b;
        background-color: #121722;
        border-radius: 6px;
        padding: 0.55rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #d8dce6;
    }
    .event-row b { color: #fbbf24; }

    div[data-testid="stExpander"] {
        background-color: #121722;
        border: 1px solid #1f2430;
        border-radius: 8px;
    }

    /* Scroll-driven reveal: subtle fade + rise, triggered natively by the
       browser as each section enters the viewport. No JS, nothing fake. */
    .reveal {
        animation: reveal-in linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 25%;
    }
    @keyframes reveal-in {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.title("EmergeRoute")
st.caption("AI-based traffic policy generation. Proposes, simulates, and ranks traffic-control actions before recommending one.")

# ---------------------------------------------------------------------------
# Top control bar (replaces the sidebar). Plain columns, inline with content.
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        sumocfg = st.selectbox(
            "Traffic scenario",
            options=["simulation.sumocfg", "simulation_heavy_stable.sumocfg", "simulation_realistic.sumocfg", "simulation_heavy.sumocfg"] + (["simulation_uploaded.sumocfg"] if __import__("pathlib").Path("simulation_uploaded.sumocfg").exists() else []),
            format_func=lambda x: {
                "simulation.sumocfg": "Light traffic",
                "simulation_heavy_stable.sumocfg": "Heavy congestion",
                "simulation_realistic.sumocfg": "Realistic (varying) traffic",
                "simulation_heavy.sumocfg": "Gridlock (worst-case)",
            }.get(x, "Uploaded video"),
        )
    with c2:
        use_prediction = st.checkbox(
            "Use AI prediction (XGBoost) to set congestion level",
            value=True,
            help="Predicts near-future congestion from recent traffic history, instead of setting it manually.",
        )
        if not use_prediction:
            congestion_level = st.select_slider(
                "Congestion level",
                options=["low", "moderate", "high"],
                value="high",
            )
        else:
            congestion_level = None
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Generate & Test Policies", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div></div>', unsafe_allow_html=True)


def stat_pill(label: str, value):
    return f'<div class="stat-pill">{label}: <b>{value}</b></div>'


PLOTLY_DARK = dict(
    paper_bgcolor="#121722",
    plot_bgcolor="#121722",
    font_color="#e6e9ef",
    margin=dict(l=10, r=10, t=40, b=10),
)

if run_button:
    if use_prediction:
        traffic_log_map = {
            "simulation.sumocfg": "traffic_log.csv",
            "simulation_heavy.sumocfg": "traffic_log_long.csv",
            "simulation_realistic.sumocfg": "traffic_log_realistic.csv",
            "simulation_heavy_stable.sumocfg": "traffic_log_heavy.csv",
        }
        log_path = traffic_log_map.get(sumocfg, "traffic_log_realistic.csv")
        with st.spinner("Training prediction model on recent traffic history..."):
            model, feature_cols = train_predictor(log_path)
            df = pd.read_csv(log_path).sort_values("time_s").reset_index(drop=True)
            recent = df.tail(N_LAGS + 1)
            congestion_level = predict_congestion_level(model, feature_cols, recent)

    with st.spinner("Probing real network conditions..."):
        probe = probe_network(sumocfg=sumocfg)

    busiest_tls = probe["ranked_tls"]
    candidates = generate_candidate_policies(busiest_tls, congestion_level=congestion_level)

    with st.spinner(f"Testing {len(candidates)} candidate policies in SUMO simulation. This may take a few minutes..."):
        ranked = rank_policies(candidates, sumocfg=sumocfg)

    scenario_label = {
        "simulation.sumocfg": "Light traffic",
        "simulation_heavy_stable.sumocfg": "Heavy congestion",
        "simulation_realistic.sumocfg": "Realistic (varying) traffic",
        "simulation_heavy.sumocfg": "Gridlock (worst-case)",
    }.get(sumocfg, "Uploaded video")
    pareto_count = sum(1 for e in ranked if e["pareto_optimal"])

    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-bar">'
        + stat_pill("Scenario", scenario_label)
        + stat_pill("Intersections monitored", len(busiest_tls))
        + stat_pill("Peak vehicles (probe)", probe["peak_vehicles"])
        + stat_pill("Jam events detected", len(probe["events"]))
        + stat_pill("Policies tested", len(candidates))
        + stat_pill("Pareto-optimal", pareto_count)
        + '</div>',
        unsafe_allow_html=True,
    )
    if use_prediction:
        st.markdown(f'<div class="status-banner ok">Predicted near-future congestion level: <b>{congestion_level.upper()}</b> (based on recent traffic trend, via XGBoost)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-banner ok">Busiest intersections (by measured queue length): {", ".join(busiest_tls[:3])}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Zone cards
    # -----------------------------------------------------------------
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("Monitored Intersections")
    top_zones = busiest_tls[:6]
    zone_cols = st.columns(3)
    max_q = max(probe["queue_totals"].values()) if probe["queue_totals"] else 1
    for i, tid in enumerate(top_zones):
        q = probe["queue_totals"][tid]
        severity = "SEVERE" if q > 0.66 * max_q else ("BUSY" if q > 0.33 * max_q else "NORMAL")
        color = "#ef4444" if severity == "SEVERE" else ("#f59e0b" if severity == "BUSY" else "#22c55e")
        with zone_cols[i % 3]:
            st.markdown(f"""
            <div class="zone-card">
                <div class="zone-id">{tid}</div>
                <div class="zone-queue" style="color:{color};">{q}</div>
                <div class="zone-label">accumulated queue, {severity}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Event feed
    # -----------------------------------------------------------------
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("Traffic Events (probe window)")
    if probe["events"]:
        for ev in probe["events"][:10]:
            st.markdown(
                f'<div class="event-row"><b>JAM</b> &mdash; vehicle <b>{ev["vehicle_id"]}</b> '
                f'teleported after waiting too long, at t={ev["time_s"]}s</div>',
                unsafe_allow_html=True,
            )
        if len(probe["events"]) > 10:
            st.caption(f"+ {len(probe['events']) - 10} more events in the full probe window.")
    else:
        st.markdown('<div class="event-row" style="border-left-color:#22c55e;">No jam events detected in the probe window &mdash; traffic flowed without major stalling.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Recommended policy
    # -----------------------------------------------------------------
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("Recommended Policy")
    top = ranked[0]
    st.markdown(f"### {top['policy_name']}")
    st.markdown(f'<div class="status-banner ok">{top["explanation"]}</div>', unsafe_allow_html=True)

    m = top["metrics"]
    cols = st.columns(6)
    with cols[0]: metric_card("Congestion delay", f"{m['avg_time_loss_s']:.0f}s")
    with cols[1]: metric_card("Avg travel time", f"{m['avg_travel_time_s']:.0f}s")
    with cols[2]: metric_card("Worst-case wait", f"{m['max_waiting_time_s']:.0f}s")
    with cols[3]: metric_card("CO2 emitted", f"{m['co2_kg']:.1f}kg")
    with cols[4]: metric_card("Fairness gap", f"{m['fairness_gap_s']:.0f}s")
    with cols[5]: metric_card("Trips completed", str(m["completed_trips"]))
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Charts -- Plotly, explicit rank order (fixes the alphabetical-sort
    # bug st.bar_chart had), plus two real donut charts.
    # -----------------------------------------------------------------
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("Policy Comparison")

    policy_order = [f"#{e['rank']} {e['policy_name']}" for e in ranked]
    comp_df = pd.DataFrame([
        {
            "Policy": f"#{e['rank']} {e['policy_name']}",
            "Congestion delay (s)": e["metrics"]["avg_time_loss_s"],
            "Avg travel time (s)": e["metrics"]["avg_travel_time_s"],
            "CO2 (kg)": e["metrics"]["co2_kg"],
        }
        for e in ranked
    ])
    comp_melted = comp_df.melt(id_vars="Policy", var_name="Metric", value_name="Value")
    fig_bar = px.bar(
        comp_melted, x="Policy", y="Value", color="Metric", barmode="group",
        category_orders={"Policy": policy_order},
    )
    fig_bar.update_layout(**PLOTLY_DARK, legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_bar, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.caption("Share of total CO2 across tested policies")
        fig_donut1 = go.Figure(data=[go.Pie(
            labels=comp_df["Policy"], values=comp_df["CO2 (kg)"], hole=0.6,
            marker=dict(line=dict(color="#0b0e14", width=2)),
        )])
        fig_donut1.update_layout(**PLOTLY_DARK, showlegend=True, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_donut1, use_container_width=True)
    with d2:
        st.caption("Share of total congestion delay across tested policies")
        fig_donut2 = go.Figure(data=[go.Pie(
            labels=comp_df["Policy"], values=comp_df["Congestion delay (s)"], hole=0.6,
            marker=dict(line=dict(color="#0b0e14", width=2)),
        )])
        fig_donut2.update_layout(**PLOTLY_DARK, showlegend=True, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_donut2, use_container_width=True)

    fair_df = pd.DataFrame([
        {
            "Policy": f"#{e['rank']} {e['policy_name']}",
            "Fairness gap (s)": e["metrics"]["fairness_gap_s"],
            "Trips completed": e["metrics"]["completed_trips"],
        }
        for e in ranked
    ])
    f1, f2 = st.columns(2)
    with f1:
        fig_fair = px.bar(fair_df, x="Policy", y="Fairness gap (s)", category_orders={"Policy": policy_order})
        fig_fair.update_traces(marker_color="#60a5fa")
        fig_fair.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig_fair, use_container_width=True)
    with f2:
        fig_trips = px.bar(fair_df, x="Policy", y="Trips completed", category_orders={"Policy": policy_order})
        fig_trips.update_traces(marker_color="#34d399")
        fig_trips.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig_trips, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # All candidates
    # -----------------------------------------------------------------
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("All Candidate Policies (ranked)")
    for entry in ranked:
        with st.expander(f"#{entry['rank']}: {entry['policy_name']}"):
            if entry["pareto_optimal"]:
                st.markdown('<span class="pareto-tag">PARETO-OPTIMAL</span>', unsafe_allow_html=True)
            st.write(entry["explanation"])
            m = entry["metrics"]
            c = st.columns(6)
            with c[0]: metric_card("Congestion delay", f"{m['avg_time_loss_s']:.0f}s")
            with c[1]: metric_card("Avg travel time", f"{m['avg_travel_time_s']:.0f}s")
            with c[2]: metric_card("Worst-case wait", f"{m['max_waiting_time_s']:.0f}s")
            with c[3]: metric_card("CO2 emitted", f"{m['co2_kg']:.1f}kg")
            with c[4]: metric_card("Fairness gap", f"{m['fairness_gap_s']:.0f}s")
            with c[5]: metric_card("Trips completed", str(m["completed_trips"]))
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Policies are tested via real SUMO simulation (not simulated results) and ranked "
        "using NSGA-II multi-objective optimization across congestion, travel time, safety, "
        "emergency delay, environmental impact, and fairness. Pareto-optimal policies "
        "represent genuine trade-offs: no other tested policy strictly beats them on every objective. "
        "Zone cards, events, and all stats above come directly from live simulation probes -- nothing shown is placeholder or fabricated."
    )

else:
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-banner ok" style="font-size:1rem;">
        Configure a scenario above and click <b>Generate & Test Policies</b> to run the full pipeline on a live SUMO simulation.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("How it works")
    steps = [
        ("01", "Probe the network", "A short live SUMO simulation measures real queue lengths at every traffic-light intersection -- no guessing which ones are busiest."),
        ("02", "Predict (optional)", "XGBoost, trained on your own logged simulation history, forecasts near-future congestion so the system can act proactively instead of only reacting."),
        ("03", "Generate candidates", "A rule-based policy generator proposes several explainable traffic-control actions: signal timing extensions, network-wide adjustments, emergency corridors."),
        ("04", "Simulate every candidate", "Each candidate policy is tested end-to-end in a full SUMO simulation run -- real results, not estimates."),
        ("05", "Score with NSGA-II", "Every candidate is ranked across six objectives at once: congestion, travel time, safety, emergency delay, emissions, and fairness."),
        ("06", "Recommend, with reasons", "The top policy is surfaced with a plain-language explanation of why it was chosen over the alternatives."),
    ]
    step_cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Step {num}</div>
                <div style="font-size:1.05rem; font-weight:600; color:#f2f4f8; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.85rem; color:#aab1c2; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    st.header("Built on")
    stack = ["SUMO", "Python", "pymoo (NSGA-II)", "XGBoost", "Streamlit", "Plotly"]
    st.markdown(
        '<div class="stat-bar">' + "".join(f'<div class="stat-pill">{s}</div>' for s in stack) + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="status-banner">
        Every number this dashboard shows after you run a scenario -- queue lengths, predicted congestion, policy scores -- comes directly from a live SUMO simulation or a model trained on your own logged data. Nothing is placeholder.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

render_video_upload()
render_detection_section()