"""Vehicle detection section for the EmergeRoute dashboard.

Plays the detected video next to per-class counts that update with the playhead.
Counts come only from assets/detections_log.json (written by the detector run).
If that file is missing, the video plays alone and the page says so.

Log format (one entry per processed frame, t in seconds of the ORIGINAL video):
[{"t": 0.017, "counts": {"car": 3, "bus": 1, "truck": 1}}, ...]
"""
import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

VIDEO = Path("assets/detected_traffic.mp4")
LOG = Path("assets/detections_log.json")

TEMPLATE = """
<style>
 body{margin:0;font-family:system-ui,sans-serif;color:#e6edf3}
 .wrap{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}
 video{width:100%;border-radius:6px;background:#000;display:block}
 .panel{background:#161b22;border:1px solid #2a3340;border-radius:6px;padding:14px 16px}
 .h{font-size:13px;color:#8b98a5;margin:0 0 10px}
 .row{display:grid;grid-template-columns:84px 1fr 34px;gap:10px;align-items:center;margin:9px 0;font-size:14px}
 .bar{height:8px;background:#232c38;border-radius:2px;overflow:hidden}
 .bar i{display:block;height:100%;width:0;background:#4da3ff;transition:width .25s ease}
 .n{text-align:right;font-variant-numeric:tabular-nums}
 .tot{display:flex;justify-content:space-between;border-top:1px solid #2a3340;margin-top:12px;padding-top:12px;font-size:14px}
 .tot b{font-size:26px;font-variant-numeric:tabular-nums}
 .sm{font-size:12px;color:#8b98a5;margin-top:10px;line-height:1.5}
 @media(max-width:700px){.wrap{grid-template-columns:1fr}}
</style>
<div class="wrap">
 <video id="v" controls muted playsinline src="data:video/mp4;base64,__VIDEO__"></video>
 <div class="panel">
  <p class="h">Detected in current frame</p>
  <div id="rows"></div>
  <div class="tot"><span>All vehicles</span><b id="tot">0</b></div>
  <p class="sm" id="sm"></p>
 </div>
</div>
<script>
const LOG=__LOG__;
const v=document.getElementById('v'),rows=document.getElementById('rows');
const classes=[...new Set(LOG.flatMap(e=>Object.keys(e.counts)))].sort();
const peak={};classes.forEach(c=>peak[c]=Math.max(...LOG.map(e=>e.counts[c]||0),1));
const allPeak=Math.max(...LOG.map(e=>Object.values(e.counts).reduce((a,b)=>a+b,0)));
classes.forEach(c=>{rows.insertAdjacentHTML('beforeend',
 `<div class="row"><span>${c}</span><div class="bar"><i id="b_${c}"></i></div><span class="n" id="n_${c}">0</span></div>`)});
document.getElementById('sm').textContent=
 'Peak in this clip: '+allPeak+' vehicles in one frame. '+LOG.length+' frames logged.';
function at(t){let lo=0,hi=LOG.length-1;while(lo<hi){const m=(lo+hi+1)>>1;LOG[m].t<=t?lo=m:hi=m-1}return LOG[lo]}
function paint(){const e=at(v.currentTime);let s=0;
 classes.forEach(c=>{const k=e.counts[c]||0;s+=k;
  document.getElementById('n_'+c).textContent=k;
  document.getElementById('b_'+c).style.width=(k/peak[c]*100)+'%'});
 document.getElementById('tot').textContent=s;
 if(!v.paused&&!v.ended)requestAnimationFrame(paint)}
v.addEventListener('play',paint);v.addEventListener('seeked',paint);v.addEventListener('timeupdate',()=>{if(v.paused)paint()});
paint();
</script>
"""


def render_detection_section():
    st.markdown("## Vehicle Detection")
    if not VIDEO.exists():
        st.info("Detected video not found at assets/detected_traffic.mp4.")
        return
    if not LOG.exists():
        st.video(str(VIDEO))
        st.caption("Per-frame counts file (assets/detections_log.json) not found, so no counts are shown.")
        return
    log = json.loads(LOG.read_text())
    video_b64 = base64.b64encode(VIDEO.read_bytes()).decode()
    html = TEMPLATE.replace("__VIDEO__", video_b64).replace("__LOG__", json.dumps(log))
    components.html(html, height=420)
    st.caption("Counts are read from the detector's own output for each frame. Nothing here is estimated.")