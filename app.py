"""
Real-Time Object Detection with YOLO26 & Streamlit
====================================================
Uses your webcam + NVIDIA GPU to run YOLO26 inference in real-time.
Toggle between Nano (speed) and Medium (accuracy) models, adjust
confidence threshold, and monitor FPS — all from a sleek Streamlit UI.
"""

import time
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO26 Real-Time Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Glow Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e 0%, #0f0c29 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* Glassmorphic Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        animation: fade-in 0.8s ease-out;
    }
    
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(123, 47, 247, 0.15) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    .hero-header h1 {
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-header p {
        position: relative;
        z-index: 1;
        color: rgba(255,255,255,0.7);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Stat cards with hover micro-animations */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .stat-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 25px rgba(0, 210, 255, 0.15);
    }
    .stat-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #a0aec0;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 10px rgba(0, 210, 255, 0.2);
    }

    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Streamlit controls styling */
    .stSlider > div[data-baseweb="slider"] {
        padding-top: 10px;
    }
    .stRadio > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02);
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
        margin-bottom: 5px;
    }
    .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(0, 210, 255, 0.3);
    }
    
    /* Smooth toggle wrapper */
    div[data-testid="stCheck"] {
        padding: 5px;
    }

    /* Video frame */
    .video-frame {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 16px 40px rgba(0,0,0,0.5);
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load Model (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str) -> YOLO:
    """Load a YOLO26 model and move it to GPU if available."""
    model = YOLO(model_name)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


# ── Helper: Draw Detections with FPS overlay ─────────────────────────────────
# Beautiful colour palette keyed by class id (cycles for > 20 classes)
_PALETTE = [
    (0, 210, 255),   # cyan
    (123, 47, 247),  # purple
    (255, 106, 193), # pink
    (0, 255, 170),   # mint
    (255, 200, 0),   # gold
    (255, 85, 85),   # coral
    (100, 220, 255), # light blue
    (180, 130, 255), # lavender
    (0, 200, 120),   # teal
    (255, 155, 80),  # orange
]


def draw_detections(frame: np.ndarray, results, fps: float) -> np.ndarray:
    """Render bounding boxes, labels, and FPS on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{result.names[cls_id]} {conf:.0%}"
            color = _PALETTE[cls_id % len(_PALETTE)]

            # Semi-transparent filled rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            overlay = frame.copy()

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── FPS Badge (top-right) ────────────────────────────────────────────────
    fps_text = f"FPS: {fps:.1f}"
    (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    pad = 10
    rx1, ry1 = w - fw - 3 * pad, 8
    rx2, ry2 = w - pad, 8 + fh + 2 * pad

    # Rounded-rect background
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (rx1 + pad, ry2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2, cv2.LINE_AA)

    return frame


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Model selector
    model_choice = st.radio(
        "🧠 Model Variant",
        options=["Nano (Speed)", "Medium (Accuracy)"],
        index=0,
        help="**Nano** is ~3× faster. **Medium** is more accurate on small / distant objects.",
    )
    model_file = "yolo26n.pt" if model_choice.startswith("Nano") else "yolo26m.pt"

    st.divider()

    # Confidence threshold
    confidence = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Only show detections above this confidence score.",
    )

    st.divider()

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.success(f"🟢 **GPU Active**\n\n{gpu_name} — {vram:.1f} GB")
    else:
        st.warning("🟡 **CPU Mode** — No CUDA GPU detected.\nInference will be slower.")

    st.divider()
    st.caption("Built with [Ultralytics](https://ultralytics.com) YOLO26 & Streamlit")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🎯 YOLO26 Real-Time Detection</h1>
    <p>End-to-end, NMS-free object detection — powered by your GPU</p>
</div>
""", unsafe_allow_html=True)

# ── Main Area ────────────────────────────────────────────────────────────────
col_video, col_stats = st.columns([3, 1])

with col_stats:
    st.markdown("### 📊 Live Stats")
    fps_placeholder = st.empty()
    detections_placeholder = st.empty()
    model_placeholder = st.empty()

    model_placeholder.markdown(f"""
    <div class="stat-card">
        <div class="label">Active Model</div>
        <div class="value" style="font-size:1rem;">{model_file.replace('.pt','').upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col_video:
    run_toggle = st.toggle("▶️  Start Detection", value=False)
    frame_placeholder = st.empty()

# ── Detection Loop ───────────────────────────────────────────────────────────
if run_toggle:
    model = load_model(model_file)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera connection.")
        st.stop()

    # Attempt to set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    st.toast(f"🚀 Detection started with **{model_file}** @ confidence ≥ {confidence:.0%}", icon="✅")

    # FPS tracking (exponential moving average)
    fps_ema = 0.0
    alpha = 0.3  # smoothing factor

    try:
        while run_toggle:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame from webcam.")
                break

            # Run YOLO26 inference
            results = model.predict(
                frame,
                conf=confidence,
                verbose=False,
                stream=False,
            )

            # Compute FPS
            dt = time.perf_counter() - t0
            instant_fps = 1.0 / max(dt, 1e-6)
            fps_ema = alpha * instant_fps + (1 - alpha) * fps_ema if fps_ema > 0 else instant_fps

            # Count detections
            n_detections = sum(len(r.boxes) for r in results if r.boxes is not None)

            # Draw on frame
            annotated = draw_detections(frame, results, fps_ema)

            # Convert BGR → RGB for Streamlit
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Update UI
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            fps_placeholder.markdown(f"""
            <div class="stat-card">
                <div class="label">FPS</div>
                <div class="value">{fps_ema:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

            detections_placeholder.markdown(f"""
            <div class="stat-card">
                <div class="label">Objects</div>
                <div class="value">{n_detections}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Detection error: {e}")
    finally:
        cap.release()

else:
    # Idle state — show a nice placeholder
    frame_placeholder.markdown("""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        height: 420px;
        background: linear-gradient(135deg, rgba(15,12,41,0.6), rgba(36,36,62,0.6));
        border-radius: 16px;
        border: 1px dashed rgba(255,255,255,0.15);
    ">
        <div style="text-align:center;">
            <p style="font-size: 3rem; margin: 0;">📷</p>
            <p style="color: rgba(255,255,255,0.4); font-size: 1rem; margin-top: 0.5rem;">
                Toggle <b>Start Detection</b> to begin
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fps_placeholder.markdown("""
    <div class="stat-card">
        <div class="label">FPS</div>
        <div class="value">—</div>
    </div>
    """, unsafe_allow_html=True)

    detections_placeholder.markdown("""
    <div class="stat-card">
        <div class="label">Objects</div>
        <div class="value">—</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    YOLO26 Real-Time Object Detection · Powered by Ultralytics & Streamlit
</div>
""", unsafe_allow_html=True)
