# EmergeRoute
### AI-Based Traffic Policy Generation System — with YDFNet Vehicle Detection

EmergeRoute is a multi-objective framework for automated urban traffic
policy generation and simulation-based evaluation. It generates multiple
candidate traffic-control actions, tests every one of them in a real SUMO
simulation before recommending anything, and scores each candidate across
six real objectives — congestion, travel time, safety, emergency delay,
environmental impact, and fairness — rather than optimizing for a single
metric.

This repository contains two integrated parts:
1. **YDFNet** — a custom vehicle detection model (perception layer)
2. **EmergeRoute Core** — the simulation, prediction, policy generation, and
   scoring pipeline (decision layer), with a live Streamlit dashboard

**Live demo:** https://ydfnet.streamlit.app

---

## System Architecture

```
Real/simulated traffic data
        ↓
XGBoost Prediction (near-future congestion forecast)
        ↓
Rule-Based Policy Generator (candidate traffic actions)
        ↓
SUMO Simulation (every candidate tested, not estimated)
        ↓
NSGA-II Scoring Engine (ranks across 6 objectives, pymoo)
        ↓
Streamlit Dashboard (recommendation + plain-language explanation)
```

Vehicle detection (YDFNet) runs as a separate, standalone perception module
that feeds real vehicle-count/density data into the system — see Part 1
below for its own details.

---

## Part 1 — YDFNet: Vehicle Detection Module

A from-scratch implementation of **YOLOv10 backbone + BiFPN neck + DETR-style
transformer detection head**, trained and evaluated on real UA-DETRAC
traffic footage.

### Architecture
```
Image → YOLOv10 backbone (P3/P4/P5) → BiFPN (multi-scale fusion) →
DETR-style head (set prediction, no NMS) → class + bbox predictions
```
- **Backbone**: pretrained YOLOv10, multi-scale feature extraction
- **Neck**: custom BiFPN (learned weighted fusion across scales)
- **Head**: DETR-style transformer decoder with learned object queries —
  trained via Hungarian bipartite matching, no NMS post-processing needed
- **Loss**: Hungarian matcher + classification / L1 / GIoU box loss

### Training & Results
Trained on 6,000 real images sampled from UA-DETRAC (82,085 total
available), 12 epochs, RTX 4050 GPU (~69 min total training time). All 12
checkpoints were systematically evaluated; **best checkpoint: `checkpoints/ydfnet_epoch5.pt`**.

Full methodology and honest findings — including a documented "query
collapse" issue in later training epochs and a cross-validation comparison
against a converged YOLOv10 reference model — are in
[`YDFNet_Vehicle_Detection_Results_Writeup.md`](./YDFNet_Vehicle_Detection_Results_Writeup.md).

### Extended Vehicle Coverage — Open-Vocabulary Detection
Standard COCO/UA-DETRAC-trained models misclassify regional vehicle types
like auto-rickshaws (no matching category in either dataset). This was
resolved using **YOLO-World** for open-vocabulary detection — specifying
`"three-wheeler auto rickshaw"` / `"tuk-tuk taxi"` as text-prompted classes
at inference time, with no additional training or labeled data required.
See `yolo_world_merged.py`.

### Running YDFNet
```bash
python demo.py --checkpoint checkpoints\ydfnet_epoch5.pt --image <path.jpg> --score-threshold 0.5
```

---

## Part 2 — EmergeRoute Core: Simulation, Prediction & Policy Engine

### What's included
- **`run_policy.py`** — TraCI-based simulation harness; applies a candidate
  policy's signal-timing changes and runs a full SUMO simulation, returning
  real metrics (congestion delay, travel time, worst-case wait, CO2)
- **`policy_generator.py`** — rule-based candidate policy generator; probes
  the network for genuinely measured congestion (real queue-length sensing
  via TraCI) rather than guessing which intersections to target
- **`log_traffic_data.py`** — logs real time-series congestion data from a
  running SUMO simulation, used as training data for prediction
- **`traffic_prediction.py`** — XGBoost regressor predicting near-future
  congestion from recent traffic history (lag features), feeding directly
  into the policy generator so the system is proactive, not just reactive
- **`scoring_engine.py`** — runs every candidate through simulation, scores
  them with NSGA-II (via `pymoo`) across all 6 objectives, returns a ranked
  list with plain-language explanations
- **`dashboard.py`** — Streamlit app tying the full pipeline together

### Network & Scenarios
- `network.net.xml` — synthetic 4x4 grid network (12 signalized
  intersections), generated via SUMO's own `netgenerate` — standard
  practice for prototype-stage traffic research
- `simulation.sumocfg` / `routes.rou.xml` — light traffic scenario
- `simulation_heavy.sumocfg` / `routes_heavy_stable.rou.xml` — heavy
  congestion scenario
- `simulation_realistic.sumocfg` / `routes_realistic.rou.xml` — realistic,
  varying-congestion scenario (used for prediction model training)
- `sumo_simulation/` — a simpler single-intersection test network, used for
  initial TraCI connectivity verification

### Verified Real Results

**NSGA-II scoring** — confirmed genuine Pareto-optimal trade-offs, not a
single arbitrary winner. Example (realistic traffic scenario): Baseline and
an "Aggressive green extension (+20s)" policy were both Pareto-optimal —
the aggressive policy reduced worst-case wait (174s vs 191s) at the cost of
slightly higher average delay and CO2.

**XGBoost prediction** — trained on 157 real simulation samples, tested on
40 held-out samples (chronological split). **Mean Absolute Error: 7.38
vehicles**, against a test-set range of 31-117 vehicles (~8.6% relative
error).

### Running the Dashboard Locally
```bash
pip install -r requirements.txt
# SUMO: download from https://sumo.dlr.de/docs/Downloads.php
export SUMO_HOME=/path/to/sumo   # or set as a Windows environment variable
streamlit run dashboard.py
```

### Deployment
Deployed on **Streamlit Community Cloud**: https://ydfnet.streamlit.app

`packages.txt` installs SUMO as a system dependency (apt-level, required
alongside `requirements.txt`'s Python packages — SUMO is a binary, not a
pip package).

---

## Full Project Structure

```
.
├── models/                          # YDFNet architecture
│   ├── backbone.py
│   ├── bifpn.py
│   ├── detr_head.py
│   ├── loss.py
│   └── ydfnet.py
├── data/                            # YDFNet dataset loaders
│   ├── ua_detrac_yolo_dataset.py
│   └── ua_detrac_dataset.py
├── checkpoints/                     # YDFNet trained weights (all 12 epochs)
├── train.py                         # YDFNet training
├── demo.py                          # YDFNet single-image inference
├── sweep_checkpoints.py             # YDFNet checkpoint comparison
├── yolo_world_merged.py             # Open-vocabulary rickshaw detection
│
├── run_policy.py                    # EmergeRoute: SUMO/TraCI simulation harness
├── policy_generator.py              # EmergeRoute: rule-based policy generator
├── scoring_engine.py                # EmergeRoute: NSGA-II scoring
├── traffic_prediction.py            # EmergeRoute: XGBoost prediction
├── log_traffic_data.py              # EmergeRoute: training data logger
├── dashboard.py                     # EmergeRoute: Streamlit dashboard
├── network.net.xml / *.rou.xml / *.sumocfg   # SUMO scenarios
├── sumo_simulation/                 # Basic single-intersection test network
│
├── requirements.txt                 # Python dependencies
├── packages.txt                     # Apt dependencies (SUMO) for deployment
└── YDFNet_Vehicle_Detection_Results_Writeup.md
```

---

## Team

| Name | Contribution |
|---|---|
| Avishi Patidar | Dataset Collection, Traffic Video Processing |
| Ayushi Kumari | YOLO Vehicle Detection & Density Estimation |
| Aditi Roy | SUMO Simulation & NSGA-II Optimization |
| Meryl Adrina Kerobin | Literature Survey, Problem Analysis, Documentation |
| Sanket Suri | XGBoost Prediction Model & Feature Engineering |
| Anant Paliwal | System Integration, Testing, Explainability Module |

---

## Datasets Used
- [UA-DETRAC](https://detrac-db.rit.albany.edu/) — vehicle detection training
- SUMO-generated synthetic traffic (`netgenerate` + `randomTrips.py`) —
  simulation scenarios, no external city dataset dependency

## Honest Scope Notes
- Simulation network is a synthetic 4x4 grid, not a real city network
  (architecture is network-size-agnostic and would scale to a real OSM-based
  network)
- One policy type (signal-timing adjustment) is fully implemented and
  demoed; diversions, one-way conversions, and emergency corridors are
  architected for via the same TraCI harness but not yet built
- Query collapse observed in later YDFNet training epochs — diagnosed as a
  known DETR-training phenomenon, addressed via systematic checkpoint
  evaluation rather than assuming more epochs is always better (see the
  results writeup for details)