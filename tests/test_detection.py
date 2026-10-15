# ──────────────────────────────────────────────
# CrowdGuard – Detection Test Script
# ──────────────────────────────────────────────

"""
End-to-end test for the PersonDetector module.

Loads a sample crowd image, runs YOLO person detection,
prints structured results, and saves an annotated copy.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_detection
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `config` / `detection` resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from detection.detector import Detection, PersonDetector


# ── Paths ────────────────────────────────────

SAMPLE_IMAGE = PROJECT_ROOT / "data" / "videos" / "test_crowd.jpg"
OUTPUT_IMAGE = PROJECT_ROOT / "data" / "videos" / "test_crowd_annotated.jpg"


def divider(title: str = "") -> None:
    """Print a visual section divider."""
    print(f"\n{'─' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 60}")


def run_test() -> None:
    # ── 1. Load image ────────────────────────
    divider("1 · Loading sample image")
    assert SAMPLE_IMAGE.exists(), f"Sample image not found: {SAMPLE_IMAGE}"

    frame = cv2.imread(str(SAMPLE_IMAGE))
    assert frame is not None, "cv2.imread returned None – corrupt image?"
    h, w = frame.shape[:2]
    print(f"  Image loaded : {SAMPLE_IMAGE.name}")
    print(f"  Resolution   : {w} × {h}")

    # ── 2. Initialise detector ───────────────
    divider("2 · Initialising PersonDetector")
    detector = PersonDetector()
    assert not detector.is_loaded, "Model should be lazy-loaded"
    print("  Detector created (model not yet loaded)")

    # ── 3. Run detection ─────────────────────
    divider("3 · Running detection")
    detections = detector.detect(frame)
    assert detector.is_loaded, "Model should be loaded after first detect()"
    print(f"  Persons found: {len(detections)}")

    # ── 4. Validate Detection objects ────────
    divider("4 · Detection results")
    for i, det in enumerate(detections):
        assert isinstance(det, Detection), "Expected Detection dataclass"
        assert det.bbox.shape == (4,), f"bbox shape mismatch: {det.bbox.shape}"
        assert 0.0 <= det.confidence <= 1.0, f"confidence out of range: {det.confidence}"
        assert det.class_name == "person", f"unexpected class: {det.class_name}"
        assert det.class_id == 0, f"unexpected class_id: {det.class_id}"

        x1, y1, x2, y2 = det.bbox
        cx, cy = det.center
        print(
            f"  [{i:>2}]  bbox=({x1:6.1f}, {y1:6.1f}, {x2:6.1f}, {y2:6.1f})  "
            f"center=({cx:6.1f}, {cy:6.1f})  "
            f"conf={det.confidence:.2%}  "
            f"class={det.class_name}"
        )

    # ── 5. Draw & save annotated image ───────
    divider("5 · Drawing annotations")
    annotated = detector.draw(frame, detections)
    assert annotated.shape == frame.shape, "Annotated shape mismatch"
    # Ensure original frame was NOT mutated
    assert not np.array_equal(annotated, frame) or len(detections) == 0, \
        "draw() should return a copy, not mutate the original"

    cv2.imwrite(str(OUTPUT_IMAGE), annotated)
    print(f"  Annotated image saved → {OUTPUT_IMAGE.name}")

    # ── 6. Summary ───────────────────────────
    divider("✅  ALL CHECKS PASSED")
    print(f"  Persons detected : {len(detections)}")
    print(f"  Model            : {detector.model_name}")
    print(f"  Confidence ≥     : {detector.confidence}")
    print(f"  Output           : {OUTPUT_IMAGE}")
    print()


if __name__ == "__main__":
    run_test()
