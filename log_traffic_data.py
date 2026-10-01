"""
EmergeRoute — Traffic Data Logger

Runs a SUMO simulation and logs network-wide congestion metrics at regular
time intervals, producing a real time-series dataset. This is the "historical
data" the prediction module trains on -- generated from our own simulation
rather than requiring an external dataset download (METR-LA etc. remain a
valid drop-in alternative later; this keeps the module fully self-contained
and demonstrable today).

Logged every `interval` seconds:
    - total vehicles currently in the network
    - average speed across all vehicles
    - total halting (queued/stopped) vehicles -- our core congestion signal

Usage:
    python log_traffic_data.py --sumocfg simulation_heavy.sumocfg --interval 30 --duration 1000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402


def log_traffic_data(sumocfg: str, interval: int = 30, duration: int = 1000, out_csv: str = "traffic_log.csv"):
    traci.start(["sumo", "-c", sumocfg, "--no-warnings", "true"])

    rows = []
    step = 0
    next_log_time = 0

    while step < duration and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step >= next_log_time:
            vehicle_ids = traci.vehicle.getIDList()
            n_vehicles = len(vehicle_ids)
            if n_vehicles > 0:
                speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                avg_speed = sum(speeds) / len(speeds)
                n_halting = sum(1 for s in speeds if s < 0.1)
            else:
                avg_speed = 0.0
                n_halting = 0

            rows.append({
                "time_s": step,
                "n_vehicles": n_vehicles,
                "avg_speed": round(avg_speed, 3),
                "n_halting": n_halting,
            })
            next_log_time += interval

        step += 1

    traci.close()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "n_vehicles", "avg_speed", "n_halting"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Logged {len(rows)} time-steps to {out_csv} (every {interval}s over {duration}s of simulation)")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, default="simulation_heavy.sumocfg")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--out", type=str, default="traffic_log.csv")
    args = parser.parse_args()

    log_traffic_data(args.sumocfg, args.interval, args.duration, args.out)
