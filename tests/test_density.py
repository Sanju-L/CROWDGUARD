# ──────────────────────────────────────────────
# CrowdGuard -- Density Analyser Unit Tests
# ──────────────────────────────────────────────

"""
Unit + integration tests for the zone-based density module.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_density
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from analytics.density import DensityAnalyser, ZoneReport, classify_density

# ── Helpers ──────────────────────────────────

PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def divider(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ── Test classify_density ────────────────────

def test_classify_density() -> None:
    divider("classify_density()")

    thresholds = {"SAFE": 0.25, "NORMAL": 0.50, "WARNING": 0.75, "HIGH": 0.90}

    check(classify_density(0.0, thresholds) == "SAFE",      "0% -> SAFE")
    check(classify_density(0.10, thresholds) == "SAFE",     "10% -> SAFE")
    check(classify_density(0.24, thresholds) == "SAFE",     "24% -> SAFE")
    check(classify_density(0.25, thresholds) == "NORMAL",   "25% -> NORMAL")
    check(classify_density(0.49, thresholds) == "NORMAL",   "49% -> NORMAL")
    check(classify_density(0.50, thresholds) == "WARNING",  "50% -> WARNING")
    check(classify_density(0.74, thresholds) == "WARNING",  "74% -> WARNING")
    check(classify_density(0.75, thresholds) == "HIGH",     "75% -> HIGH")
    check(classify_density(0.89, thresholds) == "HIGH",     "89% -> HIGH")
    check(classify_density(0.90, thresholds) == "CRITICAL", "90% -> CRITICAL")
    check(classify_density(1.00, thresholds) == "CRITICAL", "100% -> CRITICAL")
    check(classify_density(1.50, thresholds) == "CRITICAL", "150% -> CRITICAL (over capacity)")


# ── Test empty frame ─────────────────────────

def test_empty_frame() -> None:
    divider("Empty frame (no people)")

    analyser = DensityAnalyser()
    reports = analyser.compute(centroids=[], frame_shape=(480, 640))

    check(len(reports) == 4, f"4 zones returned (got {len(reports)})")
    for r in reports:
        check(r.people_count == 0, f"{r.name}: count=0")
        check(r.density_pct == 0.0, f"{r.name}: density=0%")
        check(r.status == "SAFE", f"{r.name}: status=SAFE")


# ── Test known placement ─────────────────────

def test_known_placement() -> None:
    divider("Known placement (controlled centroids)")

    zones = {
        "Zone A": {"bounds": (0.0, 0.0, 0.5, 0.5), "capacity": 10},
        "Zone B": {"bounds": (0.5, 0.0, 1.0, 0.5), "capacity": 10},
        "Zone C": {"bounds": (0.0, 0.5, 0.5, 1.0), "capacity": 10},
        "Zone D": {"bounds": (0.5, 0.5, 1.0, 1.0), "capacity": 10},
    }
    analyser = DensityAnalyser(zones=zones)

    # Place 3 people in Zone A (top-left quadrant of 640x480)
    # Zone A pixel bounds: (0,0) - (320,240)
    centroids = [
        (100.0, 100.0),   # Zone A
        (200.0, 50.0),    # Zone A
        (150.0, 200.0),   # Zone A
        (400.0, 100.0),   # Zone B
        (500.0, 300.0),   # Zone D
        (500.0, 350.0),   # Zone D
    ]

    reports = analyser.compute(centroids, frame_shape=(480, 640))
    by_name = {r.name: r for r in reports}

    check(by_name["Zone A"].people_count == 3, "Zone A: 3 people")
    check(by_name["Zone B"].people_count == 1, "Zone B: 1 person")
    check(by_name["Zone C"].people_count == 0, "Zone C: 0 people")
    check(by_name["Zone D"].people_count == 2, "Zone D: 2 people")

    # Density checks (capacity=10 per zone)
    check(by_name["Zone A"].density_pct == 30.0, "Zone A: 30%")
    check(by_name["Zone B"].density_pct == 10.0, "Zone B: 10%")
    check(by_name["Zone D"].density_pct == 20.0, "Zone D: 20%")

    # Status checks
    check(by_name["Zone A"].status == "NORMAL", "Zone A: NORMAL (30%)")
    check(by_name["Zone B"].status == "SAFE",   "Zone B: SAFE (10%)")
    check(by_name["Zone C"].status == "SAFE",   "Zone C: SAFE (0%)")
    check(by_name["Zone D"].status == "SAFE",   "Zone D: SAFE (20%)")


# ── Test critical overflow ───────────────────

def test_critical_overflow() -> None:
    divider("Critical overflow (over capacity)")

    zones = {
        "Zone A": {"bounds": (0.0, 0.0, 1.0, 1.0), "capacity": 5},
    }
    analyser = DensityAnalyser(zones=zones)

    # 8 people in a zone with capacity 5 -> 160%
    centroids = [(100.0 + i * 50, 100.0) for i in range(8)]
    reports = analyser.compute(centroids, frame_shape=(480, 640))

    check(reports[0].people_count == 8, "8 people counted")
    check(reports[0].density_pct == 160.0, "160% density")
    check(reports[0].status == "CRITICAL", "CRITICAL status")


# ── Test custom zones ────────────────────────

def test_custom_zones() -> None:
    divider("Custom zone configuration")

    zones = {
        "Left":  {"bounds": (0.0, 0.0, 0.3, 1.0), "capacity": 20},
        "Mid":   {"bounds": (0.3, 0.0, 0.7, 1.0), "capacity": 30},
        "Right": {"bounds": (0.7, 0.0, 1.0, 1.0), "capacity": 20},
    }
    analyser = DensityAnalyser(zones=zones)
    reports = analyser.compute(centroids=[], frame_shape=(480, 640))

    check(len(reports) == 3, f"3 custom zones (got {len(reports)})")
    names = [r.name for r in reports]
    check("Left" in names, "Left zone exists")
    check("Mid" in names, "Mid zone exists")
    check("Right" in names, "Right zone exists")


# ── Test pixel bounds ────────────────────────

def test_pixel_bounds() -> None:
    divider("Pixel bounds conversion")

    zones = {
        "Zone A": {"bounds": (0.0, 0.0, 0.5, 0.5), "capacity": 10},
    }
    analyser = DensityAnalyser(zones=zones)
    reports = analyser.compute(centroids=[], frame_shape=(480, 640))

    check(reports[0].bounds_px == (0, 0, 320, 240), "Correct pixel bounds (0,0,320,240)")


# ── Test draw does not crash ─────────────────

def test_draw() -> None:
    divider("Draw function (smoke test)")

    analyser = DensityAnalyser()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    centroids = [(100.0, 100.0), (400.0, 300.0)]
    reports = analyser.compute(centroids, frame_shape=(480, 640))
    annotated = analyser.draw(frame, reports)

    check(annotated.shape == frame.shape, "Output shape matches input")
    check(not np.array_equal(annotated, frame), "Frame was annotated (not blank)")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_classify_density()
    test_empty_frame()
    test_known_placement()
    test_critical_overflow()
    test_custom_zones()
    test_pixel_bounds()
    test_draw()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} DENSITY TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
