import os
import sys
import traci

SUMO_HOME = os.environ.get("SUMO_HOME")

if not SUMO_HOME:
    raise RuntimeError("SUMO_HOME is not set.")

SUMO_BINARY = os.path.join(
    SUMO_HOME,
    "bin",
    "sumo"
)

CONFIG_FILE = "intersection.sumocfg"

traci.start([
    SUMO_BINARY,
    "-c",
    CONFIG_FILE
])

print("Connected to SUMO through TraCI.")

try:

    for step in range(60):

        traci.simulationStep()

        vehicle_count = traci.vehicle.getIDCount()

        print(
            f"Step {step + 1:02d} | "
            f"Vehicles: {vehicle_count}"
        )

finally:

    traci.close()

print("SUMO simulation finished.")