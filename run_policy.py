"""
EmergeRoute — Simulation Layer

Runs the SUMO simulation with a given traffic-light policy applied via TraCI
(SUMO's real-time control API), and returns the resulting metrics. This is
the core function the Policy Generator + Scoring Engine (pymoo/NSGA-II) call
for every candidate policy they want to test.

A "policy" here is a dict describing signal timing changes to apply at one
or more intersections -- e.g. extending green time on a congested approach.
This matches the brief's "signal timing changes" candidate-action type;
diversions / heavy-vehicle restrictions / emergency corridors follow the
same pattern (modify routing/restrictions via TraCI instead of signal timing)
and can be added as additional policy "types" using this same harness.

Usage (standalone test):
    python run_policy.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# SUMO's Python API (traci) ships inside SUMO_HOME/tools
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402


@dataclass
class Policy:
    """A candidate traffic-control action.

    name: human-readable label (used in the dashboard's plain-language explanation)
    tls_overrides: {traffic_light_id: extra_green_seconds} -- extends the
        green phase duration at specific intersections. Empty dict = baseline
        (no changes), useful as the "do nothing" comparison policy.
    """
    name: str
    tls_overrides: dict[str, int] = field(default_factory=dict)


@dataclass
class SimResult:
    policy_name: str
    completed_trips: int
    avg_duration: float
    avg_waiting_time: float
    avg_time_loss: float
    total_co2: float  # kg, environmental-impact proxy
    max_waiting_time: float  # fairness proxy: worst-case single-trip wait


def run_simulation(policy: Policy, sumocfg: str = "simulation.sumocfg", sim_seconds: int = 1000) -> SimResult:
    """Runs one full simulation with the given policy applied, returns metrics.

    Uses TraCI (SUMO's step-by-step control API) rather than a static config
    file, because policies need to be applied WHILE the simulation runs
    (e.g. changing a traffic light's timing partway through) -- this is what
    lets us later add reactive policies (e.g. "detect congestion, THEN act"),
    not just fixed pre-set configurations.
    """
    traci.start([
        "sumo", "-c", sumocfg, "--no-warnings", "true",
        "--tripinfo-output", "tripinfo_temp.xml",
    ])

    # Apply the policy's traffic-light overrides once at the start.
    # (A more advanced version would apply these reactively mid-simulation
    # based on detected congestion -- this is the natural next extension.)
    for tls_id, extra_green in policy.tls_overrides.items():
        try:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            for phase in program.phases:
                if "G" in phase.state:  # only extend actual green phases
                    phase.duration += extra_green
            traci.trafficlight.setProgramLogic(tls_id, program)
        except traci.exceptions.TraCIException as e:
            print(f"  [warning] could not apply override to {tls_id}: {e}")

    completed_durations, completed_waiting, completed_time_loss = [], [], []
    total_co2 = 0.0

    step = 0
    while step < sim_seconds and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Accumulate CO2 emissions across all currently active vehicles (mg -> kg)
        for veh_id in traci.vehicle.getIDList():
            total_co2 += traci.vehicle.getCO2Emission(veh_id) / 1_000_000.0  # mg/s -> kg/s, per step (1s)

        # Collect per-trip stats for vehicles that just finished this step
        for veh_id in traci.simulation.getArrivedIDList():
            pass  # per-trip duration/waiting handled via tripinfo output instead (more reliable)

        step += 1

    traci.close()

    # Parse the tripinfo output file TraCI/SUMO wrote during the run for
    # accurate per-trip metrics (more reliable than manual accumulation above).
    import xml.etree.ElementTree as ET
    tripinfo_path = "tripinfo_temp.xml"
    if os.path.exists(tripinfo_path):
        tree = ET.parse(tripinfo_path)
        trips = tree.getroot().findall("tripinfo")
        durations = [float(t.get("duration")) for t in trips]
        waiting = [float(t.get("waitingTime")) for t in trips]
        time_loss = [float(t.get("timeLoss")) for t in trips]
    else:
        durations, waiting, time_loss = [0], [0], [0]

    return SimResult(
        policy_name=policy.name,
        completed_trips=len(durations),
        avg_duration=sum(durations) / max(len(durations), 1),
        avg_waiting_time=sum(waiting) / max(len(waiting), 1),
        avg_time_loss=sum(time_loss) / max(len(time_loss), 1),
        total_co2=total_co2,
        max_waiting_time=max(waiting) if waiting else 0.0,
    )


if __name__ == "__main__":
    # Quick standalone test: baseline (no changes) vs. one candidate policy
    baseline = Policy(name="Baseline (no changes)")
    result = run_simulation(baseline)
    print(f"\n=== {result.policy_name} ===")
    print(f"Completed trips: {result.completed_trips}")
    print(f"Avg duration: {result.avg_duration:.1f}s")
    print(f"Avg waiting time: {result.avg_waiting_time:.1f}s")
    print(f"Avg time loss: {result.avg_time_loss:.1f}s")
    print(f"Total CO2: {result.total_co2:.2f}kg")
    print(f"Max (worst-case) waiting time: {result.max_waiting_time:.1f}s")
