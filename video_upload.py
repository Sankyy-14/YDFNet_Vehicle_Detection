"""Upload box for the EmergeRoute dashboard.

Runs video_to_scenario.py on the uploaded video, converts the detected video to
mp4 for the browser, and shows the real numbers the script measured.
"""
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

SUMMARY = Path("assets/video_summary.json")


def _latest_detected_video():
    vids = [p for ext in ("*.avi", "*.mp4") for p in Path("runs/detect").glob(f"*/{ext}")]
    return max(vids, key=lambda p: p.stat().st_mtime) if vids else None


def render_video_upload():
    st.header("Analyze your own traffic video")
    st.caption("Vehicles in the video are tracked and counted, and that traffic level drives a new "
               "SUMO scenario. Works best with a fixed camera.")
    up = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
    if up is not None and st.button("Analyze video"):
        Path("uploads").mkdir(exist_ok=True)
        path = Path("uploads") / up.name
        path.write_bytes(up.getbuffer())
        with st.spinner("Detecting and tracking vehicles. This can take a few minutes..."):
            run = subprocess.run([sys.executable, "video_to_scenario.py", str(path)],
                                 capture_output=True, text=True)
        if run.returncode != 0:
            st.error("Video analysis failed.")
            st.code(run.stderr[-1500:])
            return
        src = _latest_detected_video()
        if src is not None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg = "ffmpeg"
            try:
                conv = subprocess.run([ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                       "-movflags", "+faststart", "assets/detected_traffic.mp4"],
                                      capture_output=True, text=True)
                if conv.returncode != 0:
                    st.warning("Could not convert the detected video.")
            except FileNotFoundError:
                st.warning("ffmpeg not found. Run: pip install imageio-ffmpeg")
        st.rerun()

    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text())
        c = st.columns(4)
        c[0].metric("Unique vehicles", s["unique_vehicles"])
        c[1].metric("Clip length", f"{s['duration_s']} s")
        c[2].metric("Measured flow", f"{s['measured_veh_per_hour']} veh/h")
        c[3].metric("Used in simulation", f"{s['used_veh_per_hour']} veh/h")
        if s["clamped"]:
            st.caption("The measured flow was outside the range the network can handle, so it was limited.")
        st.success("Scenario ready. Pick Uploaded video under Traffic scenario, untick AI prediction, "
                   "and click Generate & Test Policies.")