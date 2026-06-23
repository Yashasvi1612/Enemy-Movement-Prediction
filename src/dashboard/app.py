# src/dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Defense Surveillance System — Streamlit Dashboard
#
# Shows live stats, alert feed, heatmap, and charts from the pipeline.
#
# Usage:
#   streamlit run src/dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import json
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Defense Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .alert-high   { color: #ff4444; font-weight: bold; }
    .alert-medium { color: #ff8800; font-weight: bold; }
    .alert-low    { color: #ffcc00; font-weight: bold; }
    .title-text {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
    }
    .subtitle-text {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Paths ─────────────────────────────────────────────────────────────────────
ALERT_LOG_PATH  = Path("outputs/alerts/alert_log.json")
OUTPUT_VIDEO    = Path("outputs/visualizations/pipeline_output.mp4")
SNAPSHOT_DIR    = Path("outputs/visualizations")
LOSS_PLOT       = Path("outputs/visualizations/training_loss.png")


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_alerts() -> list:
    if not ALERT_LOG_PATH.exists():
        return []
    with open(ALERT_LOG_PATH, "r") as f:
        return json.load(f)


def get_snapshots() -> list:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("snapshot_*.jpg"), reverse=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title-text">🛡️ Defense Surveillance System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Transformer-Based Trajectory Prediction & Threat Analysis</p>', unsafe_allow_html=True)
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Status-Active-green", width=120)
    st.markdown("### ⚙️ Dashboard Controls")

    auto_refresh = st.toggle("Auto Refresh", value=False)
    refresh_rate = st.slider("Refresh interval (sec)", 2, 30, 5)

    st.markdown("---")
    st.markdown("### 📁 Pipeline Info")
    st.markdown(f"**Alert log:** `{ALERT_LOG_PATH}`")
    st.markdown(f"**Output video:** `{OUTPUT_VIDEO}`")

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("**Architecture:** Transformer Encoder-Decoder")
    st.markdown("**Parameters:** 353,570")
    st.markdown("**obs_len:** 8 timesteps")
    st.markdown("**pred_len:** 12 timesteps")
    st.markdown("**Dataset:** ETH/UCY (254K sequences)")

    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🚀 Run Pipeline")
    st.code("python main.py --source data/raw/test.mp4 --save", language="bash")
    st.code("python main.py --source data/raw/test.mp4 --draw-zones --save", language="bash")


# ── Load data ─────────────────────────────────────────────────────────────────
alerts = load_alerts()
snapshots = get_snapshots()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
st.markdown("### 📊 System Overview")
col1, col2, col3, col4, col5 = st.columns(5)

total_alerts  = len(alerts)
high_alerts   = sum(1 for a in alerts if a.get("severity") == "HIGH")
medium_alerts = sum(1 for a in alerts if a.get("severity") == "MEDIUM")
low_alerts    = sum(1 for a in alerts if a.get("severity") == "LOW")
unique_tracks = len(set(a.get("track_id") for a in alerts)) if alerts else 0

with col1:
    st.metric("🚨 Total Alerts",  total_alerts)
with col2:
    st.metric("🔴 HIGH",          high_alerts)
with col3:
    st.metric("🟠 MEDIUM",        medium_alerts)
with col4:
    st.metric("🟡 LOW",           low_alerts)
with col5:
    st.metric("👤 Flagged Tracks", unique_tracks)

st.markdown("---")

# ── Main content ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.5, 1])

# ── LEFT: Snapshot viewer + Alert table ───────────────────────────────────────
with left_col:

    # Latest snapshot
    st.markdown("### 🎥 Latest Snapshot")
    if snapshots:
        latest = snapshots[0]
        img = cv2.imread(str(latest))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"Snapshot: {latest.name}", use_container_width=True)
        else:
            st.info("Could not load snapshot image.")
    else:
        st.info("No snapshots yet. Run the pipeline and press 'S' to take a snapshot.")

    st.markdown("---")

    # Alert log table
    st.markdown("### 🚨 Alert Log")
    if alerts:
        df = pd.DataFrame(alerts)

        # Color severity
        def color_severity(val):
            if val == "HIGH":
                return "background-color: #3d0000; color: #ff4444"
            elif val == "MEDIUM":
                return "background-color: #2d1a00; color: #ff8800"
            elif val == "LOW":
                return "background-color: #2d2a00; color: #ffcc00"
            return ""

        display_cols = ["timestamp", "frame_id", "track_id", "alert_type", "severity", "message"]
        available    = [c for c in display_cols if c in df.columns]
        styled_df    = df[available].style.map(color_severity, subset=["severity"])

        st.dataframe(styled_df, use_container_width=True, height=300)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Alert Log (CSV)",
            data=csv,
            file_name=f"alert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No alerts yet. Run the pipeline first:\n```\npython main.py --source data/raw/test.mp4 --save\n```")


# ── RIGHT: Charts ─────────────────────────────────────────────────────────────
with right_col:

    # Alert type breakdown
    st.markdown("### 📈 Alert Breakdown")
    if alerts:
        df = pd.DataFrame(alerts)

        # Alert type pie chart
        if "alert_type" in df.columns:
            type_counts = df["alert_type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            colors = ["#ff4444", "#ff8800", "#ffcc00", "#00d4ff"]
            wedges, texts, autotexts = ax.pie(
                type_counts.values,
                labels=type_counts.index,
                autopct="%1.0f%%",
                colors=colors[:len(type_counts)],
                textprops={"color": "white", "fontsize": 9},
            )
            for autotext in autotexts:
                autotext.set_color("white")
            ax.set_title("Alert Types", color="white", fontsize=11)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Alerts over time
        st.markdown("### ⏱️ Alerts Over Time")
        if "frame_id" in df.columns:
            df["frame_id"] = df["frame_id"].astype(int)
            frame_counts   = df.groupby("frame_id").size().reset_index(name="count")

            fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor="#0e1117")
            ax2.set_facecolor("#1e2130")
            ax2.plot(frame_counts["frame_id"], frame_counts["count"],
                     color="#00d4ff", linewidth=2)
            ax2.fill_between(frame_counts["frame_id"], frame_counts["count"],
                             alpha=0.3, color="#00d4ff")
            ax2.set_xlabel("Frame", color="white", fontsize=8)
            ax2.set_ylabel("Alerts", color="white", fontsize=8)
            ax2.set_title("Alerts per Frame", color="white", fontsize=10)
            ax2.tick_params(colors="white")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#2d3250")
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")

        # Severity bar chart
        st.markdown("### 🎯 Severity Distribution")
        severity_counts = df["severity"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 3), facecolor="#0e1117")
        ax3.set_facecolor("#1e2130")
        bar_colors = {"HIGH": "#ff4444", "MEDIUM": "#ff8800", "LOW": "#ffcc00"}
        bars = ax3.bar(
            severity_counts.index,
            severity_counts.values,
            color=[bar_colors.get(s, "#888") for s in severity_counts.index]
        )
        ax3.set_title("Alerts by Severity", color="white", fontsize=10)
        ax3.tick_params(colors="white")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#2d3250")
        ax3.set_facecolor("#1e2130")
        for bar, val in zip(bars, severity_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha="center", color="white", fontsize=9)
        st.pyplot(fig3)
        plt.close(fig3)

    else:
        st.info("Charts will appear after running the pipeline.")

    st.markdown("---")

    # Training loss plot
    st.markdown("### 🧠 Model Training Loss")
    if LOSS_PLOT.exists():
        st.image(str(LOSS_PLOT), caption="Transformer Training Loss", use_container_width=True)
    else:
        st.info("Train the model first to see loss curve.")


# ── Snapshot gallery ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🖼️ Snapshot Gallery")
if snapshots:
    cols = st.columns(min(4, len(snapshots)))
    for i, (col, snap) in enumerate(zip(cols, snapshots[:4])):
        img = cv2.imread(str(snap))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            col.image(img_rgb, caption=snap.name, use_container_width=True)
else:
    st.info("No snapshots yet. Press 'S' while the pipeline is running to capture frames.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "🛡️ Defense Surveillance System | "
    "Transformer-Based Trajectory Prediction | "
    f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
    "</p>",
    unsafe_allow_html=True
)

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()