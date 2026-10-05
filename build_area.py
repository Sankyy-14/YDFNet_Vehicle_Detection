"""Build a SUMO scenario for a real area, using OpenStreetMap streets.

Usage (from the project folder):
    python build_area.py --bbox WEST SOUTH EAST NORTH [--vph 1800]

Get the four numbers from openstreetmap.org: Export, then "Manually select a
different area". They are shown as left (west), bottom (south), right (east),
top (north).

Steps: download the streets (Overpass, through SUMO's osmGet.py, retrying
across several public mirrors if one is slow or overloaded), convert them to
a SUMO network (signals guessed from OSM tags), generate traffic with
randomTrips.py, and write areas/<id>/area.sumocfg. The dashboard can then use
that file like any other scenario.

Label for results: real streets, simulated traffic. The demand level (vehicles
per hour) is a setting you choose, or comes from an analyzed video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_AREA_KM2 = 20.0  # a dense city center / a couple of connected districts.
                      # Bigger than this risks slow Overpass downloads, long
                      # netconvert/randomTrips runs, and -- the real limit --
                      # SUMO simulating the whole network per policy candidate
                      # taking too long or too much memory on a free-tier host.
MIN_AREA_KM2 = 0.1
SIM_END_S = 1000      # matches run_policy.py's default simulation length

# Public Overpass API mirrors, tried in order. One server being slow or
# overloaded used to fail the whole build; now the next mirror is tried
# automatically before giving up.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
DOWNLOAD_TIMEOUT_S = 240   # per mirror attempt -- raised alongside MAX_AREA_KM2,
                            # since a 20 km2 query returns a lot more data than
                            # the old 4 km2 cap did


def area_km2(west: float, south: float, east: float, north: float) -> float:
    lat = (south + north) / 2
    width = (east - west) * 111.32 * math.cos(math.radians(lat))
    height = (north - south) * 110.57
    return abs(width * height)


def validate_bbox(bbox) -> float:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Bounding box must be west < east and south < north, in degrees.")
    km2 = area_km2(west, south, east, north)
    if km2 > MAX_AREA_KM2:
        raise ValueError(f"Area is {km2:.1f} km2. Please choose {MAX_AREA_KM2:.0f} km2 or less.")
    if km2 < MIN_AREA_KM2:
        raise ValueError(f"Area is {km2:.2f} km2, which is too small. Choose at least {MIN_AREA_KM2} km2.")
    return km2


def area_key(bbox) -> str:
    text = ",".join(f"{v:.5f}" for v in bbox)
    return hashlib.md5(text.encode()).hexdigest()[:10]


def _tools() -> Path:
    # Falls back to /usr/share/sumo, matching every other module in this
    # project (run_policy.py, policy_generator.py, map_section.py). On
    # Streamlit Cloud's container, "apt-get install sumo sumo-tools" puts
    # SUMO there but never sets the SUMO_HOME environment variable, so
    # requiring it outright (as this used to) failed there even though SUMO
    # was correctly installed. Local Windows installs of SUMO do set
    # SUMO_HOME, which is why this only showed up after deploying.
    home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    if not Path(home).is_dir():
        raise RuntimeError(f"SUMO_HOME ({home}) does not exist. Is SUMO installed?")
    return Path(home) / "tools"


def _run(cmd, what, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what} timed out after {timeout}s.")
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{(r.stderr or r.stdout)[-1200:]}")
    return r


def _download_streets(bbox, folder: Path, tools: Path, log) -> Path:
    """Downloads the area's streets from OpenStreetMap, trying each mirror in
    OVERPASS_MIRRORS in turn. Returns the path to the downloaded OSM file.
    Raises RuntimeError only once every mirror has failed."""
    west, south, east, north = bbox
    last_error: Exception | None = None
    for i, url in enumerate(OVERPASS_MIRRORS, start=1):
        log(f"Downloading streets from OpenStreetMap (server {i}/{len(OVERPASS_MIRRORS)})...")
        # Clear any partial file a previous failed mirror left behind, so the
        # size/existence check below reflects only this attempt.
        for leftover in folder.glob("area_bbox.osm.xml*"):
            leftover.unlink(missing_ok=True)
        try:
            _run(
                [sys.executable, str(tools / "osmGet.py"), "--bbox", f"{west},{south},{east},{north}",
                 "--prefix", "area", "--output-dir", str(folder), "--url", url],
                f"Street download (server {i}/{len(OVERPASS_MIRRORS)})",
                timeout=DOWNLOAD_TIMEOUT_S,
            )
        except RuntimeError as e:
            last_error = e
            log(f"  server {i} failed ({e}); trying the next one...")
            continue
        osm_files = sorted(folder.glob("area_bbox.osm.xml*"))
        if osm_files and osm_files[0].stat().st_size >= 500:
            return osm_files[0]
        last_error = RuntimeError("that server returned no usable street data")
        log(f"  server {i} returned no street data; trying the next one...")

    raise RuntimeError(
        f"Could not download streets from any of {len(OVERPASS_MIRRORS)} OpenStreetMap servers "
        f"(last error: {last_error}). This is a problem with those public servers, not your area -- "
        "wait a minute and try again, or try a smaller area to reduce the load on the server."
    )


def _net_stats(net_path: Path) -> dict:
    edges = tls = 0
    length_m = 0.0
    for _, el in ET.iterparse(str(net_path), events=("end",)):
        if el.tag == "edge" and el.get("function") != "internal":
            edges += 1
            lane = el.find("lane")
            if lane is not None and lane.get("length"):
                length_m += float(lane.get("length"))
        elif el.tag == "tlLogic":
            tls += 1
        el.clear()
    return {"edges": edges, "signals": tls, "road_km": round(length_m / 1000, 1)}


def build_area(bbox, vph: int = 1800, out_root: str = "areas", log=print) -> dict:
    """Downloads, converts and writes a scenario. Returns a dict with the config
    path and basic stats. Cached: the same box is not downloaded twice."""
    km2 = validate_bbox(bbox)
    west, south, east, north = bbox
    key = area_key(bbox)
    folder = Path(out_root) / key
    folder.mkdir(parents=True, exist_ok=True)
    net = folder / "area.net.xml"
    tools = _tools()

    if not net.exists():
        osm_file = _download_streets(bbox, folder, tools, log)

        log("Building the road network...")
        _run(["netconvert", "--osm-files", str(osm_file), "-o", str(net),
              "--proj.utm", "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
              "--junctions.join", "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
              "--tls.default-type", "static", "--keep-edges.by-vclass", "passenger",
              "--remove-edges.isolated", "--no-warnings"], "Network conversion")

    stats = _net_stats(net)
    if stats["edges"] < 10:
        raise RuntimeError("The area has too few drivable roads. Try a larger or denser area.")

    log("Generating traffic...")
    period = 3600.0 / max(vph, 1)
    _run([sys.executable, str(tools / "randomTrips.py"), "-n", str(net),
          "-r", str(folder / "routes.rou.xml"), "-o", str(folder / "trips.trips.xml"),
          "-b", "0", "-e", str(SIM_END_S), "-p", f"{period:.3f}",
          "--fringe-factor", "5", "--min-distance", "200", "--seed", "42"], "Traffic generation")

    cfg = folder / "area.sumocfg"
    cfg.write_text(
        '<configuration>\n'
        '  <input><net-file value="area.net.xml"/><route-files value="routes.rou.xml"/></input>\n'
        f'  <time><begin value="0"/><end value="{SIM_END_S}"/></time>\n'
        '  <processing><ignore-route-errors value="true"/><time-to-teleport value="300"/></processing>\n'
        '  <report><no-warnings value="true"/></report>\n'
        '</configuration>\n')

    meta = {"key": key, "bbox": list(bbox), "area_km2": round(km2, 2), "sumocfg": str(cfg),
            "demand_veh_per_hour": vph, **stats}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    if stats["signals"] == 0:
        log("WARNING: no signalized junctions found, so signal and junction policies cannot apply. "
            "Try a denser area, such as a city center.")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    ap.add_argument("--vph", type=int, default=1800, help="vehicles per hour entering the network")
    a = ap.parse_args()
    print(json.dumps(build_area(tuple(a.bbox), vph=a.vph), indent=2))