# ──────────────────────────────────────────────
# Generate a synthetic test video from the crowd image
# by panning a crop window across it over ~90 frames.
# ──────────────────────────────────────────────

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

SRC = PROJECT_ROOT / "data" / "videos" / "test_crowd.jpg"
DST = PROJECT_ROOT / "data" / "videos" / "test_crowd.mp4"

img = cv2.imread(str(SRC))
assert img is not None, f"Cannot read {SRC}"

H, W = img.shape[:2]
# Output crop: 640 × 480, pan horizontally
crop_w, crop_h = min(640, W), min(480, H)
total_frames = 90
fps = 15

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(DST), fourcc, fps, (crop_w, crop_h))

max_x = W - crop_w
max_y = H - crop_h

for i in range(total_frames):
    # Pan right then left (ping-pong)
    t = i / max(total_frames - 1, 1)
    ping = 2 * t if t < 0.5 else 2 * (1 - t)
    x = int(ping * max_x)
    y = max_y // 2  # vertical centre

    crop = img[y : y + crop_h, x : x + crop_w].copy()

    # Add slight brightness jitter to simulate real footage
    jitter = np.random.randint(-5, 6)
    crop = np.clip(crop.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

    out.write(crop)

out.release()
print(f"[OK] Generated test video: {DST}  ({total_frames} frames, {fps} fps)")
