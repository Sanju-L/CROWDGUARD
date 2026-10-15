# ──────────────────────────────────────────────
# CrowdGuard -- Flow Analyser Unit Tests
# ──────────────────────────────────────────────

"""
Tests using synthetic point trajectories so every
expected value can be verified by hand.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_flow
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from analytics.flow import (
    FlowAnalyser,
    PersonFlow,
    ZoneFlow,
    classify_direction,
    compute_angle,
    compute_displacement,
    compute_speed,
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


# ── Test compute_displacement ────────────────

def test_displacement() -> None:
    divider("compute_displacement()")

    # Person moving right: 10px per frame for 5 frames
    hist = [(100.0 + i * 10, 200.0) for i in range(6)]
    dx, dy = compute_displacement(hist, lookback=5)
    check(approx(dx, 50.0), f"dx=50 (got {dx})")
    check(approx(dy, 0.0),  f"dy=0  (got {dy})")

    # Person moving down: 8px per frame
    hist_down = [(300.0, 100.0 + i * 8) for i in range(6)]
    dx, dy = compute_displacement(hist_down, lookback=5)
    check(approx(dx, 0.0),  f"dx=0  (got {dx})")
    check(approx(dy, 40.0), f"dy=40 (got {dy})")

    # Too short history -> (0, 0)
    dx, dy = compute_displacement([(50.0, 50.0)], lookback=5)
    check(dx == 0.0 and dy == 0.0, "single point -> (0,0)")

    # Empty history
    dx, dy = compute_displacement([], lookback=5)
    check(dx == 0.0 and dy == 0.0, "empty history -> (0,0)")

    # Lookback larger than history
    hist_short = [(0.0, 0.0), (30.0, 0.0)]
    dx, dy = compute_displacement(hist_short, lookback=10)
    check(approx(dx, 30.0), f"clamped lookback dx=30 (got {dx})")

    # Diagonal movement (right+down)
    hist_diag = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    dx, dy = compute_displacement(hist_diag, lookback=2)
    check(approx(dx, 20.0) and approx(dy, 20.0), f"diagonal (20,20) got ({dx},{dy})")


# ── Test compute_speed ───────────────────────

def test_speed() -> None:
    divider("compute_speed()")

    # 30px displacement over 5 frames at 15fps -> 5/15 = 0.333s -> 30/0.333 = 90 px/s
    s = compute_speed(30.0, 0.0, frames=5, fps=15.0)
    check(approx(s, 90.0), f"30px over 5f@15fps = 90 px/s (got {s:.1f})")

    # Diagonal: sqrt(30^2 + 40^2) = 50, over 10 frames at 10fps = 1s -> 50 px/s
    s = compute_speed(30.0, 40.0, frames=10, fps=10.0)
    check(approx(s, 50.0), f"diagonal 50px over 1s = 50 px/s (got {s:.1f})")

    # Zero frames -> 0
    s = compute_speed(10.0, 10.0, frames=0, fps=15.0)
    check(s == 0.0, "0 frames -> 0 speed")

    # Zero fps -> 0
    s = compute_speed(10.0, 10.0, frames=5, fps=0.0)
    check(s == 0.0, "0 fps -> 0 speed")

    # Stationary
    s = compute_speed(0.0, 0.0, frames=5, fps=15.0)
    check(s == 0.0, "no displacement -> 0 speed")


# ── Test compute_angle ───────────────────────

def test_angle() -> None:
    divider("compute_angle()")

    check(approx(compute_angle(10, 0), 0.0),     "right = 0 deg")
    check(approx(compute_angle(0, 10), 90.0),    "down  = 90 deg")
    check(approx(compute_angle(-10, 0), 180.0),  "left  = 180 deg")
    check(approx(compute_angle(0, -10), 270.0),  "up    = 270 deg")

    # Diagonal right+down = 45 deg
    check(approx(compute_angle(10, 10), 45.0),   "right+down = 45 deg")

    # Diagonal left+up = 225 deg
    check(approx(compute_angle(-10, -10), 225.0), "left+up = 225 deg")


# ── Test classify_direction ──────────────────

def test_classify_direction() -> None:
    divider("classify_direction()")

    threshold = 5.0

    check(classify_direction(0.0, 0.0, threshold) == "STATIONARY",   "0 speed -> STATIONARY")
    check(classify_direction(4.9, 0.0, threshold) == "STATIONARY",   "4.9 speed -> STATIONARY")
    check(classify_direction(10.0, 0.0, threshold) == "RIGHT",       "0 deg  -> RIGHT")
    check(classify_direction(10.0, 90.0, threshold) == "DOWN",       "90 deg -> DOWN")
    check(classify_direction(10.0, 180.0, threshold) == "LEFT",      "180 deg -> LEFT")
    check(classify_direction(10.0, 270.0, threshold) == "UP",        "270 deg -> UP")
    check(classify_direction(10.0, 44.9, threshold) == "RIGHT",      "44.9 deg -> RIGHT")
    check(classify_direction(10.0, 45.0, threshold) == "DOWN",       "45 deg   -> DOWN")
    check(classify_direction(10.0, 134.9, threshold) == "DOWN",      "134.9 deg -> DOWN")
    check(classify_direction(10.0, 135.0, threshold) == "LEFT",      "135 deg   -> LEFT")
    check(classify_direction(10.0, 315.0, threshold) == "RIGHT",     "315 deg   -> RIGHT")
    check(classify_direction(10.0, 350.0, threshold) == "RIGHT",     "350 deg   -> RIGHT")


# ── Test FlowAnalyser: person flows ─────────

def test_person_flows() -> None:
    divider("FlowAnalyser.compute_person_flows()")

    analyser = FlowAnalyser(fps=10.0, lookback=5, stationary_threshold=5.0)

    # Person 1: moving right at 20px/frame
    # Person 2: stationary
    # Person 3: moving up at 10px/frame
    histories = {
        1: [(100.0 + i * 20, 200.0) for i in range(6)],
        2: [(300.0, 300.0)] * 6,
        3: [(400.0, 300.0 - i * 10) for i in range(6)],
    }

    flows = analyser.compute_person_flows(histories)
    by_id = {pf.track_id: pf for pf in flows}

    # Person 1: dx=100, dy=0, distance=100, time=0.5s, speed=200
    check(by_id[1].direction == "RIGHT", "Person 1: RIGHT")
    check(approx(by_id[1].speed, 200.0), f"Person 1: speed ~200 (got {by_id[1].speed})")

    # Person 2: dx=0, dy=0, speed=0
    check(by_id[2].direction == "STATIONARY", "Person 2: STATIONARY")
    check(by_id[2].speed == 0.0, f"Person 2: speed=0 (got {by_id[2].speed})")

    # Person 3: dx=0, dy=-50, moving up, speed=100
    check(by_id[3].direction == "UP", "Person 3: UP")
    check(approx(by_id[3].speed, 100.0), f"Person 3: speed ~100 (got {by_id[3].speed})")


# ── Test FlowAnalyser: zone flows ───────────

def test_zone_flows() -> None:
    divider("FlowAnalyser.compute_zone_flows()")

    zones = {
        "Zone A": {"bounds": (0.0, 0.0, 0.5, 0.5), "capacity": 50},
        "Zone B": {"bounds": (0.5, 0.0, 1.0, 0.5), "capacity": 50},
    }
    analyser = FlowAnalyser(fps=10.0, lookback=5, stationary_threshold=5.0, zones=zones)

    # 3 people in Zone A (top-left), 1 in Zone B
    person_flows = [
        PersonFlow(track_id=1, dx=20, dy=0, speed=40.0, angle_deg=0, direction="RIGHT", center=(100.0, 100.0)),
        PersonFlow(track_id=2, dx=15, dy=0, speed=30.0, angle_deg=0, direction="RIGHT", center=(200.0, 100.0)),
        PersonFlow(track_id=3, dx=0, dy=0, speed=0.0, angle_deg=0, direction="STATIONARY", center=(150.0, 200.0)),
        PersonFlow(track_id=4, dx=-20, dy=0, speed=40.0, angle_deg=180, direction="LEFT", center=(500.0, 100.0)),
    ]

    zone_flows = analyser.compute_zone_flows(person_flows, frame_shape=(480, 640))
    by_name = {zf.zone_name: zf for zf in zone_flows}

    # Zone A: 3 people, dominant=RIGHT (2 right, 1 stationary)
    za = by_name["Zone A"]
    check(za.total_count == 3, f"Zone A: 3 people (got {za.total_count})")
    check(za.moving_count == 2, f"Zone A: 2 moving (got {za.moving_count})")
    check(za.stationary_count == 1, f"Zone A: 1 stationary (got {za.stationary_count})")
    check(za.dominant_direction == "RIGHT", f"Zone A: dominant=RIGHT (got {za.dominant_direction})")
    expected_avg = (40 + 30 + 0) / 3
    check(approx(za.average_speed, expected_avg), f"Zone A: avg speed ~{expected_avg:.1f} (got {za.average_speed})")

    # Zone B: 1 person moving left
    zb = by_name["Zone B"]
    check(zb.total_count == 1, f"Zone B: 1 person (got {zb.total_count})")
    check(zb.dominant_direction == "LEFT", f"Zone B: dominant=LEFT (got {zb.dominant_direction})")


# ── Test empty inputs ────────────────────────

def test_empty() -> None:
    divider("Empty inputs")

    analyser = FlowAnalyser(fps=15.0)

    flows = analyser.compute_person_flows({})
    check(len(flows) == 0, "No histories -> no person flows")

    zone_flows = analyser.compute_zone_flows([], frame_shape=(480, 640))
    for zf in zone_flows:
        check(zf.dominant_direction == "STATIONARY", f"{zf.zone_name}: STATIONARY with no people")
        check(zf.total_count == 0, f"{zf.zone_name}: 0 people")


# ── Test draw smoke test ────────────────────

def test_draw() -> None:
    divider("Draw function (smoke test)")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person_flows = [
        PersonFlow(track_id=1, dx=30, dy=0, speed=60.0, angle_deg=0, direction="RIGHT", center=(200.0, 200.0)),
    ]
    zone_flows = [
        ZoneFlow(zone_name="Zone A", dominant_direction="RIGHT", average_speed=60.0, moving_count=1, stationary_count=0, total_count=1),
    ]

    annotated = FlowAnalyser.draw(frame, person_flows, zone_flows)
    check(annotated.shape == frame.shape, "Output shape matches")
    check(not np.array_equal(annotated, frame), "Frame was annotated")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_displacement()
    test_speed()
    test_angle()
    test_classify_direction()
    test_person_flows()
    test_zone_flows()
    test_empty()
    test_draw()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} FLOW TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
