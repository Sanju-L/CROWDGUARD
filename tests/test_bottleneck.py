# ──────────────────────────────────────────────
# CrowdGuard -- Bottleneck Detector Unit Tests
# ──────────────────────────────────────────────

"""
Tests using synthetic zone transitions with hand-verifiable
rate calculations.

Formulas verified:
    inflow_rate       = crossings_a_to_b / window_seconds * 60
    outflow_rate      = crossings_b_to_a / window_seconds * 60
    accumulation_rate = inflow_rate - outflow_rate
    bottleneck        = |accumulation_rate| >= threshold

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_bottleneck
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from analytics.bottleneck import (
    BottleneckDetector,
    GateReport,
    assign_zone,
)

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


def approx(a: float, b: float, tol: float = 0.5) -> bool:
    return abs(a - b) < tol


def divider(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# Simple 2-zone setup for clear testing
SIMPLE_ZONES = {
    "Left":  {"bounds": (0.0, 0.0, 0.5, 1.0), "capacity": 50},
    "Right": {"bounds": (0.5, 0.0, 1.0, 1.0), "capacity": 50},
}

SIMPLE_GATES = {
    "Gate_LR": {
        "zone_a": "Left", "zone_b": "Right",
        "line": (0.5, 0.0, 0.5, 1.0),
    },
}


# ── Test assign_zone ─────────────────────────

def test_assign_zone() -> None:
    divider("assign_zone()")

    zones = SIMPLE_ZONES

    z = assign_zone(100.0, 200.0, (480, 640), zones)
    check(z == "Left", f"(100,200) -> Left (got {z})")

    z = assign_zone(400.0, 200.0, (480, 640), zones)
    check(z == "Right", f"(400,200) -> Right (got {z})")

    # Edge: exactly at boundary of Right (x=320)
    z = assign_zone(320.0, 200.0, (480, 640), zones)
    check(z == "Right", f"(320,200) -> Right (got {z})")

    # Outside all zones
    z = assign_zone(700.0, 200.0, (480, 640), zones)
    check(z is None, f"(700,200) -> None (got {z})")


# ── Test: no crossings ───────────────────────

def test_no_crossings() -> None:
    divider("No crossings (static)")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # All people stay in Left zone for 5 frames
    for f in range(5):
        detector.update({1: "Left", 2: "Left"}, frame_number=f)

    reports = detector.get_reports()
    r = reports[0]

    check(r.crossings_a_to_b == 0, "0 crossings a->b")
    check(r.crossings_b_to_a == 0, "0 crossings b->a")
    check(r.inflow_rate == 0.0, "inflow = 0")
    check(r.outflow_rate == 0.0, "outflow = 0")
    check(r.accumulation_rate == 0.0, "accumulation = 0")
    check(r.bottleneck is False, "No bottleneck")


# ── Test: balanced crossings ─────────────────

def test_balanced_crossings() -> None:
    divider("Balanced crossings (no bottleneck)")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # Frame 0: person 1 in Left, person 2 in Right
    detector.update({1: "Left", 2: "Right"}, frame_number=0)
    # Frame 1: they swap
    detector.update({1: "Right", 2: "Left"}, frame_number=1)

    reports = detector.get_reports()
    r = reports[0]

    # 1 crossing L->R, 1 crossing R->L
    check(r.crossings_a_to_b == 1, f"1 crossing a->b (got {r.crossings_a_to_b})")
    check(r.crossings_b_to_a == 1, f"1 crossing b->a (got {r.crossings_b_to_a})")

    # Rates: 1/10*60 = 6 p/min each
    check(approx(r.inflow_rate, 6.0), f"inflow ~6 (got {r.inflow_rate})")
    check(approx(r.outflow_rate, 6.0), f"outflow ~6 (got {r.outflow_rate})")
    check(approx(r.accumulation_rate, 0.0), f"accumulation ~0 (got {r.accumulation_rate})")
    check(r.bottleneck is False, "Balanced -> no bottleneck")


# ── Test: bottleneck detected ────────────────

def test_bottleneck_detected() -> None:
    divider("Bottleneck detected (heavy one-way flow)")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # Simulate 5 people crossing Left -> Right on consecutive frames
    # Initial: all in Left
    zones_f0 = {i: "Left" for i in range(1, 6)}
    detector.update(zones_f0, frame_number=0)

    # Frame 1-5: each person crosses one per frame
    for f in range(1, 6):
        zones = {}
        for i in range(1, 6):
            if i <= f:
                zones[i] = "Right"
            else:
                zones[i] = "Left"
        detector.update(zones, frame_number=f)

    reports = detector.get_reports()
    r = reports[0]

    # 5 crossings L->R, 0 R->L
    check(r.crossings_a_to_b == 5, f"5 crossings a->b (got {r.crossings_a_to_b})")
    check(r.crossings_b_to_a == 0, f"0 crossings b->a (got {r.crossings_b_to_a})")

    # inflow = 5/10*60 = 30 p/min
    check(approx(r.inflow_rate, 30.0), f"inflow ~30 (got {r.inflow_rate})")
    check(approx(r.outflow_rate, 0.0), f"outflow ~0 (got {r.outflow_rate})")
    check(approx(r.accumulation_rate, 30.0), f"accumulation ~30 (got {r.accumulation_rate})")
    check(r.bottleneck is True, "30 > 10 -> BOTTLENECK")
    check("BOTTLENECK" in r.explanation, "Explanation mentions BOTTLENECK")
    print(f"    Explanation: {r.explanation}")


# ── Test: reverse bottleneck ─────────────────

def test_reverse_bottleneck() -> None:
    divider("Reverse bottleneck (outflow > inflow)")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # 4 people cross Right -> Left
    zones_f0 = {i: "Right" for i in range(1, 5)}
    detector.update(zones_f0, frame_number=0)

    for f in range(1, 5):
        zones = {}
        for i in range(1, 5):
            zones[i] = "Left" if i <= f else "Right"
        detector.update(zones, frame_number=f)

    reports = detector.get_reports()
    r = reports[0]

    # 0 a->b, 4 b->a
    check(r.crossings_b_to_a == 4, f"4 crossings b->a (got {r.crossings_b_to_a})")
    # accumulation = 0 - 24 = -24
    check(r.accumulation_rate < 0, f"negative accumulation (got {r.accumulation_rate})")
    check(r.bottleneck is True, "|accumulation| >= 10 -> BOTTLENECK")


# ── Test: window pruning ─────────────────────

def test_window_pruning() -> None:
    divider("Rolling window pruning")

    # window = 5 seconds, fps = 10 -> window = 50 frames
    detector = BottleneckDetector(
        fps=10.0, window_seconds=5.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # Crossing at frame 0
    detector.update({1: "Left"}, frame_number=0)
    detector.update({1: "Right"}, frame_number=1)

    reports = detector.get_reports()
    check(reports[0].crossings_a_to_b == 1, "1 crossing before pruning")

    # Jump to frame 100 (well past window)
    detector.update({1: "Right"}, frame_number=100)
    reports = detector.get_reports()
    check(reports[0].crossings_a_to_b == 0, "0 crossings after pruning (frame 100)")


# ── Test: custom threshold ───────────────────

def test_custom_threshold() -> None:
    divider("Custom threshold")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=50.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    # 3 crossings -> 18 p/min < 50 threshold
    detector.update({1: "Left", 2: "Left", 3: "Left"}, frame_number=0)
    detector.update({1: "Right", 2: "Right", 3: "Right"}, frame_number=1)

    reports = detector.get_reports()
    r = reports[0]

    check(approx(r.accumulation_rate, 18.0), f"accumulation ~18 (got {r.accumulation_rate})")
    check(r.bottleneck is False, "18 < 50 -> no bottleneck")
    check(r.threshold == 50.0, f"threshold=50 (got {r.threshold})")


# ── Test: reset ──────────────────────────────

def test_reset() -> None:
    divider("Reset clears state")

    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
        gates=SIMPLE_GATES, zones=SIMPLE_ZONES,
    )

    detector.update({1: "Left"}, frame_number=0)
    detector.update({1: "Right"}, frame_number=1)
    detector.reset()

    reports = detector.get_reports()
    check(reports[0].crossings_a_to_b == 0, "0 crossings after reset")


# ── Test: draw smoke test ────────────────────

def test_draw() -> None:
    divider("Draw function (smoke test)")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reports = [
        GateReport(
            gate_name="Gate_LR", zone_a="Left", zone_b="Right",
            crossings_a_to_b=5, crossings_b_to_a=1,
            inflow_rate=30.0, outflow_rate=6.0,
            accumulation_rate=24.0, bottleneck=True,
            threshold=10.0, explanation="test",
        ),
    ]

    annotated = BottleneckDetector.draw(frame, reports, gates=SIMPLE_GATES)
    check(annotated.shape == frame.shape, "Output shape matches")
    check(not np.array_equal(annotated, frame), "Frame was annotated")


# ── Test: multiple gates ─────────────────────

def test_multiple_gates() -> None:
    divider("Multiple gates (4-zone layout)")

    # Use default 4-zone/4-gate config
    detector = BottleneckDetector(
        fps=10.0, window_seconds=10.0, threshold=10.0,
    )

    # Person 1: Zone A -> Zone B (crosses Gate_AB)
    # Person 2: Zone A -> Zone C (crosses Gate_AC)
    detector.update({1: "Zone A", 2: "Zone A"}, frame_number=0)
    detector.update({1: "Zone B", 2: "Zone C"}, frame_number=1)

    reports = detector.get_reports()
    by_name = {r.gate_name: r for r in reports}

    check(by_name["Gate_AB"].crossings_a_to_b == 1, "Gate_AB: 1 crossing")
    check(by_name["Gate_AC"].crossings_a_to_b == 1, "Gate_AC: 1 crossing")
    check(by_name["Gate_CD"].crossings_a_to_b == 0, "Gate_CD: 0 crossings")
    check(by_name["Gate_BD"].crossings_a_to_b == 0, "Gate_BD: 0 crossings")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_assign_zone()
    test_no_crossings()
    test_balanced_crossings()
    test_bottleneck_detected()
    test_reverse_bottleneck()
    test_window_pruning()
    test_custom_threshold()
    test_reset()
    test_draw()
    test_multiple_gates()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} BOTTLENECK TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
