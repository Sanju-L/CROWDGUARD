# ──────────────────────────────────────────────
# CrowdGuard – Configuration
# ──────────────────────────────────────────────

"""
Central configuration for the CrowdGuard system.
All tuneable parameters live here so every module
can import a single source of truth.
"""

from pathlib import Path

# ── Project Paths ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
MAP_DIR = DATA_DIR / "maps"
MODEL_DIR = BASE_DIR / "models"

# ── Detection ─────────────────────────────────
YOLO_MODEL_NAME: str = "yolo11s.pt"          # stronger bundled model for small/distant people
DETECTION_CONFIDENCE: float = 0.12           # retain low-confidence people in dense scenes
DETECTION_IOU_THRESHOLD: float = 0.50        # merge overlapping tile detections without suppressing neighbours
DETECTION_IMAGE_SIZE: int = 640               # preserve detail for distant people
DETECTION_FRAME_INTERVAL: int = 2             # infer every other frame; reuse tracks between scans
PERSON_CLASS_ID: int = 0                     # COCO class-id for "person"

# ── Tracking ──────────────────────────────────
TRACK_BUFFER: int = 30                       # frames to keep lost tracks
MATCH_THRESHOLD: float = 0.8                 # association threshold

# ── Analytics ─────────────────────────────────
DENSITY_GRID_ROWS: int = 8                   # rows in the density grid
DENSITY_GRID_COLS: int = 8                   # columns in the density grid
FLOW_HISTORY_SECONDS: float = 2.0            # look-back window for flow vectors
ANOMALY_SPEED_SIGMA: float = 2.5             # σ-multiplier for speed anomalies
COUNTER_FLOW_ANGLE_DEG: float = 135.0        # angle threshold for counter-flow

# ── Zone-Based Density ────────────────────────
# Each zone is defined as (x1, y1, x2, y2) in *fractional*
# coordinates (0.0–1.0) relative to the frame dimensions.
#
#   Zone A | Zone B
#   ───────┼───────
#   Zone C | Zone D
#
DENSITY_ZONES: dict = {
    "Zone A": {"bounds": (0.0, 0.0, 0.5, 0.5), "capacity": 50},
    "Zone B": {"bounds": (0.5, 0.0, 1.0, 0.5), "capacity": 50},
    "Zone C": {"bounds": (0.0, 0.5, 0.5, 1.0), "capacity": 50},
    "Zone D": {"bounds": (0.5, 0.5, 1.0, 1.0), "capacity": 50},
}

# Density classification thresholds (fraction of capacity).
# NOTE: These are *prototype* values, NOT universal safety standards.
DENSITY_THRESHOLDS: dict = {
    "SAFE":     0.25,    # 0%  – 25%
    "NORMAL":   0.50,    # 25% – 50%
    "WARNING":  0.75,    # 50% – 75%
    "HIGH":     0.90,    # 75% – 90%
    #                      90%+  → CRITICAL
}

# Colours for each status level  (BGR for OpenCV)
DENSITY_STATUS_COLORS: dict = {
    "SAFE":     (0, 200, 0),       # green
    "NORMAL":   (200, 200, 0),     # cyan-ish
    "WARNING":  (0, 180, 255),     # orange
    "HIGH":     (0, 80, 255),      # red-orange
    "CRITICAL": (0, 0, 255),       # red
}

# ── Bottleneck / Virtual Gates ────────────────
# Each gate sits on the boundary between two adjacent zones.
#   "zone_a" / "zone_b"  : the two zones the gate connects
#   "line"               : gate line as (x1, y1, x2, y2) fractional coords
#   "capacity_per_min"   : maximum sustainable throughput (people/min)
#
#   Default 2×2 layout gates:
#
#        Gate_AB (vertical, top)
#     Zone A  |  Zone B
#   Gate_AC ──┼── Gate_BD
#     Zone C  |  Zone D
#        Gate_CD (vertical, bottom)
#
BOTTLENECK_GATES: dict = {
    "Gate_AB": {
        "zone_a": "Zone A", "zone_b": "Zone B",
        "line": (0.5, 0.0, 0.5, 0.5),
    },
    "Gate_CD": {
        "zone_a": "Zone C", "zone_b": "Zone D",
        "line": (0.5, 0.5, 0.5, 1.0),
    },
    "Gate_AC": {
        "zone_a": "Zone A", "zone_b": "Zone C",
        "line": (0.0, 0.5, 0.5, 0.5),
    },
    "Gate_BD": {
        "zone_a": "Zone B", "zone_b": "Zone D",
        "line": (0.5, 0.5, 1.0, 0.5),
    },
}

# Rolling window for rate calculation
BOTTLENECK_WINDOW_SECONDS: float = 10.0   # seconds of history for rate calc

# If net accumulation rate exceeds this (people/min), flag bottleneck
BOTTLENECK_ACCUMULATION_THRESHOLD: float = 10.0   # people/min

# ── Risk Scoring ──────────────────────────────
# Transparent weighted formula:
#   risk_score = sum(weight_i * normalised_input_i) for all factors
# Each input is normalised to 0–100 before combining.
#
# NOTE: These are *prototype* scoring weights and bands,
#       NOT universal safety standards.
RISK_WEIGHTS: dict = {
    "density":          0.35,
    "counter_flow":     0.20,
    "bottleneck":       0.20,
    "flow_instability": 0.15,
    "exit_congestion":  0.10,
}

# Prototype risk bands (score 0–100)
RISK_LEVELS: dict = {
    "SAFE":     (0, 39),
    "WARNING":  (40, 59),
    "HIGH":     (60, 79),
    "CRITICAL": (80, 100),
}

RISK_ALERT_THRESHOLD: float = 60.0          # score >= this triggers an alert

# ── Routing ───────────────────────────────────
DEFAULT_MAP_FILE: str = "venue_map.json"     # graph definition in data/maps/
NODE_TYPES: list = ["entrance", "zone", "exit"]

# Dynamic cost multipliers applied to base_cost based on zone risk level.
# Higher multiplier = route avoids that zone.
ROUTE_COST_MULTIPLIERS: dict = {
    "SAFE":     1,
    "WARNING":  3,
    "HIGH":     8,
    "CRITICAL": 15,
    "BLOCKED":  float("inf"),
}

# ── Dashboard / UI ──────────────────────────────
DASHBOARD_REFRESH_RATE: float = 0.5          # seconds between UI refreshes
DASHBOARD_TITLE: str = "CrowdGuard – Crowd Safety Monitor"
