# =============================================================================
#  config.py — Smart Exam Proctoring System (YOLO-only build)
# =============================================================================

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ── Input ─────────────────────────────────────────────────────────────
INPUT_SOURCE = 0   # ✅ 0 = default webcam (Mac)

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Model paths ───────────────────────────────────────────────────────
YOLO_DETECT_MODEL = "yolov8n.pt"
YOLO_POSE_MODEL   = "yolov8n-pose.pt"

# ── Detection thresholds ──────────────────────────────────────────────
DETECT_CONF = 0.5
POSE_CONF   = 0.5

# ── Object detection (cheating objects) ───────────────────────────────
ALERT_OBJECTS = [
    "cell phone",
    "book",
    "laptop",
    "remote"
]

# ── Multi-person detection ────────────────────────────────────────────
MAX_PERSONS = 1   # >1 person = alert

# ── Head pose thresholds ──────────────────────────────────────────────
HEAD_YAW_THRESH   = 25   # left/right
HEAD_PITCH_THRESH = 20   # up/down

# ── Logging ───────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "events.txt")

# ── Performance tuning ────────────────────────────────────────────────
YOLO_EVERY = 2
POSE_EVERY = 2

# ── Overlay colors (BGR) ──────────────────────────────────────────────
COLOR_OK    = (0, 210, 0)
COLOR_WARN  = (0, 200, 255)
COLOR_ALERT = (0, 0, 255)