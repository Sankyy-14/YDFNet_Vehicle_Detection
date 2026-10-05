"""
EmergeRoute - Real-Area Map Section

Lets the user search for, or draw, a real place; builds a SUMO scenario for
its actual street layout via build_area.py; runs the existing policy pipeline
on that area; and draws, on real map tiles:
  - the real street network, straight from the generated net file
  - the recommended policy's diversion, as arrows on the real edges it
    actually reroutes traffic away from (not illustrative placement)
  - a user-picked emergency route: the real shortest path through the actual
    road network between two points the user clicks

Nothing here is placed by hand. The streets, the diversion edges, and the
route all come from parsing area.net.xml, the same network file the
simulation runs on.

Label for results, matching build_area.py: real streets, simulated traffic.
The vehicle demand is a setting you choose, not something measured at the
place.

Needs two packages not already in requirements.txt: folium, streamlit-folium.
Needs "requests" (stdlib-adjacent, nearly always already present; add it to
requirements.txt too if it's not there).

Map widgets use use_container_width=True (streamlit-folium >= 0.13). If that
raises a TypeError on an older install, run:
    pip install -U streamlit-folium
or replace use_container_width=True with a fixed width=1100 on each
st_folium(...) call below.

Not yet run against a live SUMO install in this build -- test against a real
small area (for example a dense part of a city center, <= 4 km2, with
several marked traffic signals) and report back anything sumolib rejects;
exact method names (getShortestPath, convertXY2LonLat) can vary slightly
across SUMO versions.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import requests
import streamlit as st
import folium
from folium.plugins import Draw, PolyLineTextPath
from streamlit_folium import st_folium

from build_area import build_area, area_km2, MAX_AREA_KM2

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import sumolib  # noqa: E402

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ARROW_GLYPH = "   ►   "

# Map sizes. Width is handled by use_container_width=True on every st_folium
# call, so these heights are the only thing that controls how big the maps
# look.
DRAW_MAP_HEIGHT = 480
AREA_MAP_HEIGHT = 650
ROUTE_MAP_HEIGHT = 600


# ---------------------------------------------------------------------------
# Place search
# ---------------------------------------------------------------------------
def geocode_place(name: str):
    """Looks up a place name on OpenStreetMap. Returns (west, south, east,
    north) or None if nothing was found or the lookup failed."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": name, "format": "json", "limit": 1},
            headers={"User-Agent": "EmergeRoute-student-project/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
    except requests.RequestException:
        return None
    if not results:
        return None
    south, north, west, east = (float(v) for v in results[0]["boundingbox"])
    return (west, south, east, north)


def _clamp_to_max_area(bbox, max_km2: float = MAX_AREA_KM2):
    """Shrinks an oversized box to its centre instead of just rejecting it,
    so a search for a whole city still produces something build_area.py will
    accept."""
    west, south, east, north = bbox
    km2 = area_km2(west, south, east, north)
    if km2 <= max_km2:
        return bbox, False
    shrink = (max_km2 / km2) ** 0.5
    cx, cy = (west + east) / 2, (south + north) / 2
    half_w, half_h = (east - west) * shrink / 2, (north - south) * shrink / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h), True


# ---------------------------------------------------------------------------
# Network loading + geometry helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_net(net_path: str):
    return sumolib.net.readNet(net_path)


def _edge_latlon_shape(net, edge) -> list[list[float]]:
    """An edge's real shape, as [lat, lon] pairs for folium."""
    pts = []
    for x, y in edge.getShape():
        lon, lat = net.convertXY2LonLat(x, y)
        pts.append([lat, lon])
    return pts


def _point_segment_distance(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _nearest_edge(net, lat: float, lon: float):
    """The drivable edge whose real shape passes closest to a clicked point."""
    x, y = net.convertLonLat2XY(lon, lat)
    best, best_d = None, float("inf")
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        shape = edge.getShape()
        for i in range(len(shape) - 1):
            d = _point_segment_distance((x, y), shape[i], shape[i + 1])
            if d < best_d:
                best, best_d = edge, d
    return best


def _shortest_route_latlon(net, from_edge, to_edge):
    """The real shortest path for an emergency vehicle through this area's
    actual road network. Returns [[lat, lon], ...] or None if no route
    exists or this SUMO build's sumolib does not expose getShortestPath."""
    try:
        result = net.getShortestPath(from_edge, to_edge, vClass="emergency")
    except TypeError:
        result = net.getShortestPath(from_edge, to_edge)
    except AttributeError:
        return None
    edges = result[0] if isinstance(result, tuple) else result
    if not edges:
        return None
    points: list[list[float]] = []
    for e in edges:
        pts = _edge_latlon_shape(net, e)
        points.extend(pts if not points else pts[1:])
    return points


def _draw_streets(m, net):
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        folium.PolyLine(_edge_latlon_shape(net, edge), color="#5b8def", weight=2, opacity=0.45).add_to(m)


def _draw_diversion(m, net, divert_tls: list[str]):
    """Highlights, as arrows, the real approach edges a diversion policy
    actually reroutes traffic away from."""
    for tls_id in divert_tls:
        node = net.getNode(tls_id)
        if node is None:
            continue
        for edge in node.getIncoming():
            pts = _edge_latlon_shape(net, edge)
            line = folium.PolyLine(
                pts, color="#f59e0b", weight=5, opacity=0.9,
                tooltip=f"Diverted approach to junction {tls_id}",
            ).add_to(m)
            PolyLineTextPath(line, ARROW_GLYPH, repeat=True, offset=6,
                              attributes={"fill": "#f59e0b", "font-weight": "bold"}).add_to(m)


# ---------------------------------------------------------------------------
# Main section
# ---------------------------------------------------------------------------
def render_map_section():
    st.header("Real-Area Traffic Map")
    st.caption(
        "Search a real place, build a live traffic scenario for its actual street "
        "layout, and see the recommended diversion and a real emergency route on it. "
        "Real streets, simulated traffic: the road network is real, the vehicle "
        "demand below is a setting you choose."
    )

    st.session_state.setdefault("area_meta", None)
    st.session_state.setdefault("area_candidates", None)
    st.session_state.setdefault("area_top_policy", None)
    st.session_state.setdefault("emg_points", [None, None])
    st.session_state.setdefault("search_bbox", None)

    c1, c2 = st.columns([3, 1])
    with c1:
        place = st.text_input("Search a place (a city centre or neighborhood works best)", key="place_search")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search", use_container_width=True, key="place_search_btn")

    vph = st.slider("Simulated demand (vehicles per hour entering the area)", 200, 3000, 1800, step=100)

    if search_clicked and place.strip():
        with st.spinner("Looking up the place on OpenStreetMap..."):
            found = geocode_place(place.strip())
        if found is None:
            st.error("Could not find that place. Try a more specific search, or draw a box on the map below.")
            st.session_state.search_bbox = None
        else:
            bbox, shrunk = _clamp_to_max_area(found)
            st.session_state.search_bbox = bbox
            if shrunk:
                st.info(f"That place is larger than the {MAX_AREA_KM2:.0f} km2 limit, so the box was shrunk to its centre.")

    # Kept in session_state (not a local variable) so it survives the rerun
    # triggered by clicking "Build this area" itself -- otherwise the search
    # result would vanish on that click and nothing would get built.
    pending_bbox = st.session_state.search_bbox

    st.caption('Or draw a box directly on the map below, using the rectangle tool, then click "Build this area."')
    if pending_bbox:
        west, south, east, north = pending_bbox
        draw_map = folium.Map(tiles="OpenStreetMap")
        draw_map.fit_bounds([[south, west], [north, east]])
    else:
        draw_map = folium.Map(location=[23.2599, 77.4126], zoom_start=12, tiles="OpenStreetMap")
    Draw(
        export=False, position="topleft",
        draw_options={"rectangle": True, "polygon": False, "circle": False,
                      "circlemarker": False, "marker": False, "polyline": False},
        edit_options={"edit": False},
    ).add_to(draw_map)
    drawn = st_folium(
        draw_map, height=DRAW_MAP_HEIGHT, use_container_width=True,
        key="draw_map", returned_objects=["all_drawings"],
    )

    drawn_bbox = None
    if drawn and drawn.get("all_drawings"):
        coords = drawn["all_drawings"][-1]["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        drawn_bbox = (min(lons), min(lats), max(lons), max(lats))

    # A hand-drawn box means the user is actively adjusting the area, so it
    # takes priority over an earlier search result once one exists.
    build_bbox = drawn_bbox or pending_bbox
    if build_bbox:
        west, south, east, north = build_bbox
        st.write(f"Selected box: west {west:.4f}, south {south:.4f}, east {east:.4f}, north {north:.4f}")
        if st.button("Build this area", type="primary", key="build_area_btn"):
            with st.spinner("Downloading real streets and building the scenario (this can take a minute)..."):
                try:
                    meta = build_area(build_bbox, vph=vph)
                    st.session_state.area_meta = meta
                    st.session_state.area_candidates = None
                    st.session_state.area_top_policy = None
                    st.session_state.emg_points = [None, None]
                    st.session_state.search_bbox = None
                except (RuntimeError, ValueError) as e:
                    # validate_bbox (too large / too small) raises ValueError;
                    # the download/convert/traffic-generation steps raise
                    # RuntimeError. Both are a bad box or a bad run, not a
                    # bug, so both get a plain message instead of crashing.
                    st.error(str(e))

    meta = st.session_state.area_meta
    if not meta:
        return

    st.success(
        f"Built: {meta['area_km2']} km2, {meta['edges']} roads, {meta['signals']} signalized "
        f"junctions, {meta['road_km']} km of road."
    )
    if meta["signals"] == 0:
        st.warning("No signalized junctions were found here, so signal-based and diversion policies have nothing to act on. Try a denser area, such as a city center.")

    # Everything below depends on sumolib's geo-conversion (pyproj) and on
    # parsing the generated network. A failure here (missing pyproj, a
    # network with no embedded projection, etc.) must not take down the rest
    # of the dashboard -- "How it works," the policy generator, and the
    # video section all run later in dashboard.py's single top-to-bottom
    # script, so an uncaught error here would otherwise stop the whole page
    # partway through, as if those sections had been removed.
    try:
        net_path = str(Path(meta["sumocfg"]).parent / "area.net.xml")
        net = _load_net(net_path)
        west, south, east, north = meta["bbox"]
        center_lat, center_lon = (south + north) / 2, (west + east) / 2
    except Exception as e:
        st.error(f"Could not load this area's road network for the map ({e}).")
        return

    # ---- run the existing policy pipeline on this real area --------------
    if st.button("Find the best policy for this area", key="run_area_policy"):
        from policy_generator import probe_network, generate_candidate_policies
        from scoring_engine import rank_policies
        try:
            with st.spinner("Probing this area's real network and testing candidate policies..."):
                probe = probe_network(sumocfg=meta["sumocfg"])
                candidates = generate_candidate_policies(probe["ranked_tls"], congestion_level="high")
                ranked = rank_policies(candidates, sumocfg=meta["sumocfg"])
            st.session_state.area_candidates = candidates
            st.session_state.area_top_policy = ranked[0]
        except Exception as e:
            st.error(f"Could not test policies on this area ({e}).")

    top_policy = st.session_state.area_top_policy
    candidates = st.session_state.area_candidates
    divert_tls: list[str] = []
    if top_policy and candidates:
        matched = next((c for c in candidates if c.name == top_policy["policy_name"]), None)
        if matched:
            divert_tls = matched.divert_tls
        st.markdown(f"**Recommended for this area: {top_policy['policy_name']}**")
        st.caption(top_policy["explanation"])

    # ---- street + diversion map -------------------------------------------
    # pts is read BEFORE building the map so an already-confirmed start/end
    # gets a real pin drawn on it -- without this, there was no visual
    # confirmation of what you'd actually selected.
    pts = st.session_state.emg_points
    try:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
        _draw_streets(m, net)
        if divert_tls:
            _draw_diversion(m, net, divert_tls)
        elif top_policy:
            st.caption("The recommended policy here is not a diversion, so no diversion arrows are drawn.")
        if pts[0]:
            folium.Marker([pts[0]["lat"], pts[0]["lng"]], tooltip="Emergency route start",
                          icon=folium.Icon(color="green", icon="play")).add_to(m)
        if pts[1]:
            folium.Marker([pts[1]["lat"], pts[1]["lng"]], tooltip="Emergency route end",
                          icon=folium.Icon(color="red", icon="stop")).add_to(m)

        clicked = st_folium(
            m, height=AREA_MAP_HEIGHT, use_container_width=True,
            key="area_map", returned_objects=["last_clicked"],
        )
    except Exception as e:
        st.error(f"Could not draw the street map for this area ({e}).")
        return

    if clicked and clicked.get("last_clicked"):
        lc = clicked["last_clicked"]
        st.caption(
            f"Last point clicked on the map: {lc['lat']:.5f}, {lc['lng']:.5f}. "
            "Use the buttons below to set it as the emergency route's start or end -- "
            "it will show as a pin on the map above once confirmed."
        )
    else:
        st.caption("Click a point on the map above, then use the buttons below to set it as the emergency route's start or end.")

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        if st.button("Set as start", key="set_start") and clicked and clicked.get("last_clicked"):
            pts[0] = clicked["last_clicked"]
            st.session_state.emg_points = pts
            st.rerun()  # redraw immediately so the pin appears without waiting for another click
    with pcol2:
        if st.button("Set as end", key="set_end") and clicked and clicked.get("last_clicked"):
            pts[1] = clicked["last_clicked"]
            st.session_state.emg_points = pts
            st.rerun()
    with pcol3:
        if st.button("Clear points", key="clear_points"):
            st.session_state.emg_points = [None, None]
            st.rerun()

    if pts[0] and pts[1]:
        try:
            start_edge = _nearest_edge(net, pts[0]["lat"], pts[0]["lng"])
            end_edge = _nearest_edge(net, pts[1]["lat"], pts[1]["lng"])
            if not start_edge or not end_edge:
                st.warning("Could not match one of those points to a road in this area.")
            else:
                route = _shortest_route_latlon(net, start_edge, end_edge)
                if not route:
                    st.warning("No route could be found between those two points on this network.")
                else:
                    st.markdown("**Best suitable emergency route**")
                    st.caption("The real shortest path through this area's actual road network, for an emergency vehicle.")
                    rm = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
                    _draw_streets(rm, net)
                    folium.Marker([pts[0]["lat"], pts[0]["lng"]], tooltip="Start", icon=folium.Icon(color="green")).add_to(rm)
                    folium.Marker([pts[1]["lat"], pts[1]["lng"]], tooltip="End", icon=folium.Icon(color="red")).add_to(rm)
                    route_line = folium.PolyLine(route, color="#ef4444", weight=5, opacity=0.95,
                                                  tooltip="Shortest real route for an emergency vehicle").add_to(rm)
                    PolyLineTextPath(route_line, ARROW_GLYPH, repeat=True, offset=6,
                                      attributes={"fill": "#ef4444", "font-weight": "bold"}).add_to(rm)
                    st_folium(
                        rm, height=ROUTE_MAP_HEIGHT, use_container_width=True,
                        key="route_map", returned_objects=[],
                    )
        except Exception as e:
            st.error(f"Could not compute the emergency route ({e}).")