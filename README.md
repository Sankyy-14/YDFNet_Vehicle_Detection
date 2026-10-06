# EmergeRoute

### AI-based traffic policy generation, with YDFNet vehicle detection

EmergeRoute generates several candidate traffic-control policies for a congested road network, tests every one of them in a SUMO simulation, and ranks them on six objectives before recommending one. The goal is to replace one-metric, unvalidated decisions with options that were simulated and compared first.

**Live demo:** <https://ydfnet.streamlit.app>

The repository has two integrated parts:

1. **YDFNet:** a vehicle detection model (perception layer).
2. **EmergeRoute Core:** prediction, policy generation, simulation and scoring (decision layer), with a Streamlit dashboard.

---

## How it works

```
Traffic video (optional)  ->  YOLO-World detection and tracking  ->  traffic demand level
                                                                           |
Probe network (real SUMO queue lengths)  <---------------------------------+
        |
XGBoost congestion forecast (optional)
        |
Rule-based policy generator  ->  7 candidate policies
        |
SUMO simulation of every candidate (full run each)
        |
NSGA-II scoring across 6 objectives  ->  ranked list
        |
Dashboard: recommended policy, reason, comparison with baseline
```

Every number on the dashboard comes from a simulation run, a trained model, or detector output. Nothing is placeholder data.

---

## Candidate policies

| Policy type | What it does |
|---|---|
| Baseline | No changes. The reference every other policy is compared against. |
| Signal timing | Extended green (+10s) and aggressive green extension (+20s) at the 3 busiest intersections, and a network-wide +5s extension. |
| Heavy-vehicle restriction | Trucks are banned from the approach lanes of the 3 busiest intersections. |
| Diversion | Approaches to the 3 busiest intersections get a routing penalty, so vehicles reroute around them. |
| Emergency corridor | Signals switch to green ahead of an approaching emergency vehicle. |

To make the truck and emergency policies measurable, every run (baseline included) receives the same seeded extra traffic: one truck every 30 seconds and three emergency vehicles. All policies therefore face identical demand.

## Objectives

| Objective | How it is measured |
|---|---|
| Congestion | Average time lost per trip |
| Travel time | Average trip duration |
| Safety | Emergency-braking and collision events reported by SUMO |
| Emergency delay | Average time lost by the injected emergency vehicles |
| Environmental impact | Total CO2 emitted (kg) |
| Fairness | Worst-case wait minus average wait |

---

## Dashboard

- **Scenario selection:** light, heavy, realistic and gridlock scenarios, plus an uploaded-video scenario.
- **Monitored intersections:** measured queue length per intersection, severity-coded.
- **Traffic events:** real jam and teleport events from the probe.
- **Recommended policy:** shown as condition, action and reason, with eight metric cards and a table of the change against baseline. Each block is tagged simulated, predicted or derived. If no change beats the baseline, the dashboard says so.
- **Policy comparison:** one small chart per metric, with the recommended policy highlighted.
- **Video analysis:** upload a traffic video to count vehicles and build a SUMO scenario from the measured traffic level.
- **Vehicle detection panel:** the detected video with live per-class counts, a vehicle-mix chart and a vehicles-in-view chart.

Recommendations are decision support. They are validated in simulation and are not applied to real traffic signals.

---

## Part 1: YDFNet vehicle detection

A from-scratch implementation of a **YOLOv10 backbone, BiFPN neck and DETR-style transformer head**, trained on real UA-DETRAC footage.

```
Image -> YOLOv10 backbone (P3/P4/P5) -> BiFPN (multi-scale fusion) ->
DETR-style head (set prediction, no NMS) -> class and box predictions
```

- **Backbone:** pretrained YOLOv10, multi-scale feature extraction
- **Neck:** custom BiFPN with learned weighted fusion across scales
- **Head:** DETR-style decoder with learned object queries, trained with Hungarian matching and no NMS post-processing
- **Loss:** Hungarian matcher with classification, L1 and GIoU box loss

**Training:** 6,000 images sampled from UA-DETRAC (82,085 available), 12 epochs on an RTX 4050 (about 69 minutes). All 12 checkpoints were evaluated, and the best is `checkpoints/ydfnet_epoch5.pt`. Later epochs showed query collapse, a known DETR training issue, so checkpoints were compared instead of assuming more epochs is better.

**Regional vehicles:** UA-DETRAC and COCO have no auto-rickshaw category. Open-vocabulary detection with **YOLO-World** handles them through text prompts ("three-wheeler auto rickshaw", "tuk-tuk taxi") with no extra training. See `yolo_world_merged.py`.

```
python demo.py --checkpoint checkpoints\ydfnet_epoch5.pt --image <path.jpg> --score-threshold 0.5
```

## Video to scenario

Uploading a video in the dashboard runs `video_to_scenario.py`:

1. YOLO-World with ByteTrack tracks vehicles, and each unique vehicle is counted once.
2. The count becomes a traffic level in vehicles per hour, limited to 600 to 9000.
3. SUMO traffic is generated at that level on the existing network, and a new scenario file is written.
4. The detector's per-frame counts are saved for the detection panel.

Notes: the flow estimate is only meaningful for a fixed camera and a clip of at least 30 seconds. One camera's flow is used as whole-network demand, which is a simplification. AI prediction is switched off for uploaded scenarios because the model was trained on the other scenarios' logs.

---

## Part 2: EmergeRoute core

| File | Role |
|---|---|
| `run_policy.py` | TraCI harness: applies a policy, injects trucks and emergency vehicles, runs a full SUMO simulation, returns metrics |
| `policy_generator.py` | Probes the network for measured queue lengths and generates the candidate policies |
| `scoring_engine.py` | Simulates every candidate and ranks them with NSGA-II (`pymoo`) across the six objectives, with plain-language explanations |
| `traffic_prediction.py` | XGBoost regressor forecasting near-future congestion from lagged traffic history |
| `log_traffic_data.py` | Logs time-series data from a running simulation for prediction training |
| `video_to_scenario.py` | Converts a traffic video into detection counts and a SUMO scenario |
| `video_upload.py` | Dashboard upload box for video analysis |
| `detection_panel.py` | Detection video, live counts and charts |
| `dashboard.py` | Streamlit app tying everything together |

### Network and scenarios

- `network.net.xml`: synthetic 4x4 grid with 12 signalized intersections, generated with SUMO's `netgenerate`
- `simulation.sumocfg`: light traffic
- `simulation_heavy_stable.sumocfg`: heavy congestion (stable plateau, about 7,200 vehicles per hour)
- `simulation_realistic.sumocfg`: realistic, varying traffic (used for prediction training)
- `simulation_heavy.sumocfg`: gridlock, the worst case
- `sumo_simulation/`: single-intersection test network used to verify TraCI

### Verified results

These come from an earlier run with the original five signal-timing policies. Current runs test seven policies, and their results appear in the dashboard each time.

- **NSGA-II:** Baseline and the aggressive +20s green extension were both Pareto-optimal. The extension cut worst-case wait (174s against 191s) at the cost of slightly higher average delay and CO2.
- **XGBoost:** trained on 157 simulation samples and tested on 40 held-out samples with a chronological split. Mean absolute error was 7.38 vehicles on a test range of 31 to 117 (about 8.6% relative error).

---

## Run locally

```
pip install -r requirements.txt
# Install SUMO: https://sumo.dlr.de/docs/Downloads.php
# Set SUMO_HOME to the SUMO folder (a Windows environment variable, or export on Linux/macOS)
streamlit run dashboard.py
```

Video analysis uses the GPU (`device=0`). On a machine without a CUDA GPU, change that setting in `video_to_scenario.py` to run on the CPU, which is much slower.

## Deployment

Deployed on **Streamlit Community Cloud**. `packages.txt` installs SUMO as a system dependency, and `requirements.txt` lists the Python packages, including `imageio-ffmpeg` for video conversion.

---

## Project structure

```
.
├── models/                  # YDFNet architecture (backbone, BiFPN, DETR head, loss)
├── data/                    # YDFNet dataset loaders
├── checkpoints/             # YDFNet weights
├── assets/                  # Detected demo video and per-frame detection counts
├── train.py                 # YDFNet training
├── demo.py                  # YDFNet single-image inference
├── sweep_checkpoints.py     # Checkpoint comparison
├── yolo_world_merged.py     # Open-vocabulary rickshaw detection
│
├── run_policy.py            # SUMO/TraCI simulation harness
├── policy_generator.py      # Rule-based policy generator
├── scoring_engine.py        # NSGA-II scoring
├── traffic_prediction.py    # XGBoost prediction
├── log_traffic_data.py      # Training data logger
├── video_to_scenario.py     # Video to SUMO scenario
├── video_upload.py          # Dashboard upload box
├── detection_panel.py       # Detection video and charts
├── dashboard.py             # Streamlit dashboard
├── network.net.xml, *.rou.xml, *.sumocfg    # SUMO scenarios
├── sumo_simulation/         # Single-intersection test network
│
├── requirements.txt
└── packages.txt             # Apt dependencies (SUMO) for deployment
```

---

## Team

| Name | Contribution |
|---|---|
| Avishi Patidar | Dataset Collection, Traffic Video Processing |
| Ayushi Kumari | YOLO Vehicle Detection and Density Estimation |
| Aditi Roy | SUMO Simulation and NSGA-II Optimization |
| Meryl Adrina Kerobin | Literature Survey, Problem Analysis, Documentation |
| Sanket Suri | XGBoost Prediction Model and Feature Engineering |
| Anant Paliwal | System Integration, Testing, Explainability Module |

## Datasets

- [UA-DETRAC](https://detrac-db.rit.albany.edu/): vehicle detection training
- SUMO-generated synthetic traffic (`netgenerate` and `randomTrips.py`): simulation scenarios, with no external city dataset

---

## Honest scope notes

- The network is a synthetic 4x4 grid, not a real city. The architecture does not depend on network size, and importing a real OpenStreetMap area is planned future work.
- Traffic is simulated. For uploaded videos it is calibrated to the vehicle count measured in the video, so results describe "simulation calibrated to the traffic seen in this video" and not the place in the video.
- The final ranking order uses an equal-weight sum of the normalized objectives. NSGA-II identifies which policies are Pareto-optimal, meaning no other tested policy beats them on every objective.
- Green-extension policies lengthen every green phase at a junction equally. A policy that shifts green time toward the congested approach is planned.
- Safety is measured from SUMO's emergency-braking and collision events, which can be zero in light traffic.
- Query collapse appeared in later YDFNet epochs and was handled by systematic checkpoint evaluation, not by training longer.

## Future work

- Real-street networks imported from OpenStreetMap, with results drawn on a map
- Directional green-priority policy
- Faster simulation runs through TraCI subscriptions for CO2 reading
- Detection from fixed-camera footage for traffic-level calibration
