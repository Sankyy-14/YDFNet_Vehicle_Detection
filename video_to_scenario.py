"""Turn a traffic video into a SUMO scenario that the EmergeRoute dashboard can run.

Usage (from the project folder):
    python video_to_scenario.py "path\\to\\video.mp4"

Steps: track vehicles in the video, count each unique vehicle once, convert the
count to a traffic demand level, then build routes_uploaded.rou.xml and
simulation_uploaded.sumocfg by copying your simulation_realistic.sumocfg.
Also writes assets/detections_log.json for the detection panel.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

CONF = 0.25
CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'three-wheeler auto rickshaw', 'tuk-tuk taxi']
MERGE = {'three-wheeler auto rickshaw': 'auto rickshaw', 'tuk-tuk taxi': 'auto rickshaw'}
NET = "network.net.xml"
TEMPLATE_CFG = "simulation_realistic.sumocfg"
MIN_VPH, MAX_VPH = 600, 9000   # keeps demand in a range the grid network can handle


def main(source: str, scale: float = 1.0):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    model = YOLO('yolov8s-world.pt')
    model.set_classes(CLASSES)

    seen = {}          # track id -> class name (first seen)
    log = []
    for i, r in enumerate(model.track(source, stream=True, persist=True, conf=CONF,
                                      tracker="bytetrack.yaml", save=True, device='cpu')):
        counts = {}
        ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for tid, c in zip(ids, r.boxes.cls.tolist()):
            name = MERGE.get(CLASSES[int(c)], CLASSES[int(c)])
            counts[name] = counts.get(name, 0) + 1
            if tid is not None and tid not in seen:
                seen[int(tid)] = name
        log.append({"t": round(i / fps, 4), "counts": counts})

    duration = len(log) / fps
    unique = len(seen)
    vph_raw = unique / duration * 3600 * scale
    vph = int(min(MAX_VPH, max(MIN_VPH, vph_raw)))
    by_class = {}
    for n in seen.values():
        by_class[n] = by_class.get(n, 0) + 1

    Path("assets").mkdir(exist_ok=True)
    Path("assets/detections_log.json").write_text(json.dumps(log))
    summary = {"source": os.path.basename(source), "duration_s": round(duration, 1),
               "unique_vehicles": unique, "by_class": by_class,
               "measured_veh_per_hour": round(vph_raw), "used_veh_per_hour": vph,
               "clamped": vph != int(vph_raw)}
    Path("assets/video_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- build the SUMO scenario ----
    period = 3600 / vph
    cfg_text = Path(TEMPLATE_CFG).read_text()
    m = re.search(r'<end value="([\d.]+)"', cfg_text)
    end = m.group(1) if m else "600"
    tools = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo")) / "tools" / "randomTrips.py"
    subprocess.run([sys.executable, str(tools), "-n", NET, "-r", "routes_uploaded.rou.xml",
                    "-o", "trips_uploaded.trips.xml", "-b", "0", "-e", end,
                    "-p", f"{period:.3f}"], check=True)
    cfg_text = re.sub(r'<route-files value="[^"]*"', '<route-files value="routes_uploaded.rou.xml"', cfg_text)
    Path("simulation_uploaded.sumocfg").write_text(cfg_text)

    print(json.dumps(summary, indent=2))
    print("Scenario ready: simulation_uploaded.sumocfg")
    print("Detected video saved under runs/detect/ (convert to mp4 for the dashboard).")
    if duration < 30:
        print("WARNING: clip is under 30 s, so the demand estimate is very rough.")
    print("NOTE: demand is only meaningful if the camera was fixed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python video_to_scenario.py "video.mp4"')
    main(sys.argv[1])
