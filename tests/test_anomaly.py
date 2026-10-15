# ──────────────────────────────────────────────
# CrowdGuard -- Counter-Flow Detector Unit Tests
# ──────────────────────────────────────────────

"""
Tests using synthetic zone flow data with hand-verifiable
expected outcomes.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_anomaly
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.anomaly import (
    CounterFlowDetector,
    CounterFlowReport,
    DEFAULT_COUNTER_FLOW_THRESHOLD,
    OPPOSITE_DIRECTION,
)
from analytics.flow import ZoneFlow

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


def make_zone_flow(
    name: str = "Zone A",
    dominant: str = "RIGHT",
    dist: dict | None = None,
    avg_speed: float = 50.0,
    moving: int = 10,
    stationary: int = 2,
) -> ZoneFlow:
    """Helper to build a ZoneFlow with sensible defaults."""
    return ZoneFlow(
        zone_name=name,
        dominant_direction=dominant,
        direction_distribution=dist or {},
        average_speed=avg_speed,
        moving_count=moving,
        stationary_count=stationary,
        total_count=moving + stationary,
    )


# ── Test opposite direction lookup ───────────

def test_opposite_directions() -> None:
    divider("OPPOSITE_DIRECTION lookup")

    check(OPPOSITE_DIRECTION["LEFT"] == "RIGHT",  "LEFT  <-> RIGHT")
    check(OPPOSITE_DIRECTION["RIGHT"] == "LEFT",  "RIGHT <-> LEFT")
    check(OPPOSITE_DIRECTION["UP"] == "DOWN",     "UP    <-> DOWN")
    check(OPPOSITE_DIRECTION["DOWN"] == "UP",     "DOWN  <-> UP")


# ── Test: no counter-flow ────────────────────

def test_no_counter_flow() -> None:
    divider("No counter-flow (all same direction)")

    zf = make_zone_flow(
        dominant="RIGHT",
        dist={"RIGHT": 8, "DOWN": 2},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    # opposite = LEFT, count = 0, pct = 0%
    check(r.counter_flow is False, "No counter-flow flagged")
    check(r.opposing_pct == 0.0, f"opposing_pct=0.0 (got {r.opposing_pct})")
    check(r.opposite_direction == "LEFT", f"opposite=LEFT (got {r.opposite_direction})")
    check("Normal flow" in r.explanation, f"Explanation mentions 'Normal flow'")
    print(f"    Explanation: {r.explanation}")


# ── Test: counter-flow detected ──────────────

def test_counter_flow_detected() -> None:
    divider("Counter-flow detected")

    # 10 moving: 6 RIGHT, 4 LEFT -> 4/10 = 40% opposing
    zf = make_zone_flow(
        dominant="RIGHT",
        dist={"RIGHT": 6, "LEFT": 4},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is True, "Counter-flow flagged")
    check(r.opposing_pct == 40.0, f"opposing_pct=40.0 (got {r.opposing_pct})")
    check(r.opposite_count == 4, f"opposite_count=4 (got {r.opposite_count})")
    check(r.dominant_count == 6, f"dominant_count=6 (got {r.dominant_count})")
    check("Counter-flow detected" in r.explanation, "Explanation mentions detection")
    check("40%" in r.explanation, "Explanation includes 40%")
    check("4/10" in r.explanation, "Explanation includes 4/10 ratio")
    print(f"    Explanation: {r.explanation}")


# ── Test: exactly at threshold ───────────────

def test_at_threshold() -> None:
    divider("Exactly at threshold boundary")

    # 10 moving: 7 RIGHT, 3 LEFT -> 3/10 = 30% = threshold
    zf = make_zone_flow(
        dominant="RIGHT",
        dist={"RIGHT": 7, "LEFT": 3},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is True, "At threshold -> flagged (>=)")
    check(r.opposing_pct == 30.0, f"opposing_pct=30.0 (got {r.opposing_pct})")


# ── Test: just below threshold ───────────────

def test_below_threshold() -> None:
    divider("Just below threshold")

    # 10 moving: 8 DOWN, 2 UP -> 2/10 = 20% < 30%
    zf = make_zone_flow(
        dominant="DOWN",
        dist={"DOWN": 8, "UP": 2},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is False, "Below threshold -> not flagged")
    check(r.opposing_pct == 20.0, f"opposing_pct=20.0 (got {r.opposing_pct})")
    check(r.opposite_direction == "UP", f"opposite=UP (got {r.opposite_direction})")


# ── Test: all stationary ─────────────────────

def test_all_stationary() -> None:
    divider("All stationary (no movement)")

    zf = make_zone_flow(
        dominant="STATIONARY",
        dist={"STATIONARY": 5},
        moving=0,
        stationary=5,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is False, "Stationary -> no counter-flow")
    check(r.opposing_pct == 0.0, "opposing_pct=0.0")
    check("stationary" in r.explanation.lower(), "Explanation mentions stationary")
    print(f"    Explanation: {r.explanation}")


# ── Test: empty zone ─────────────────────────

def test_empty_zone() -> None:
    divider("Empty zone (no people)")

    zf = make_zone_flow(
        dominant="STATIONARY",
        dist={},
        moving=0,
        stationary=0,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is False, "Empty -> no counter-flow")
    check(r.opposing_pct == 0.0, "opposing_pct=0.0")


# ── Test: custom threshold ───────────────────

def test_custom_threshold() -> None:
    divider("Custom threshold (50%)")

    # 10 moving: 6 LEFT, 4 RIGHT -> 4/10 = 40% < 50%
    zf = make_zone_flow(
        dominant="LEFT",
        dist={"LEFT": 6, "RIGHT": 4},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=50.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is False, "40% < 50% threshold -> not flagged")
    check(r.threshold == 50.0, f"threshold=50 (got {r.threshold})")


# ── Test: UP/DOWN pair ───────────────────────

def test_up_down_counter_flow() -> None:
    divider("UP/DOWN counter-flow")

    # 8 moving: 5 UP, 3 DOWN -> 3/8 = 37.5%
    zf = make_zone_flow(
        dominant="UP",
        dist={"UP": 5, "DOWN": 3},
        moving=8,
    )
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check(r.counter_flow is True, "37.5% >= 30% -> flagged")
    check(r.dominant_direction == "UP", "dominant=UP")
    check(r.opposite_direction == "DOWN", "opposite=DOWN")
    check(r.opposite_count == 3, f"opposite_count=3 (got {r.opposite_count})")


# ── Test: multiple zones ─────────────────────

def test_multiple_zones() -> None:
    divider("Multiple zones at once")

    zones = [
        make_zone_flow("Zone A", "RIGHT", {"RIGHT": 9, "LEFT": 1}, moving=10),
        make_zone_flow("Zone B", "LEFT",  {"LEFT": 4, "RIGHT": 6}, moving=10),
        make_zone_flow("Zone C", "DOWN",  {"DOWN": 5, "UP": 5},    moving=10),
    ]
    detector = CounterFlowDetector(threshold=30.0)
    reports = detector.analyse(zones)
    by_name = {r.zone_name: r for r in reports}

    check(by_name["Zone A"].counter_flow is False, "Zone A: 10% -> no flag")
    check(by_name["Zone B"].counter_flow is True,  "Zone B: 60% -> flagged")
    check(by_name["Zone C"].counter_flow is True,  "Zone C: 50% -> flagged")


# ── Test: explanation content ────────────────

def test_explanation_content() -> None:
    divider("Explanation content validation")

    zf = make_zone_flow(
        name="Zone X",
        dominant="RIGHT",
        dist={"RIGHT": 3, "LEFT": 7},
        moving=10,
    )
    detector = CounterFlowDetector(threshold=25.0)
    reports = detector.analyse([zf])
    r = reports[0]

    check("Zone X" in r.explanation, "Explanation includes zone name")
    check("Counter-flow detected" in r.explanation, "Explanation includes detection phrase")
    check("70%" in r.explanation, "Explanation includes percentage")
    check("7/10" in r.explanation, "Explanation includes count ratio")
    check("LEFT" in r.explanation, "Explanation includes opposing direction")
    check("RIGHT" in r.explanation, "Explanation includes dominant direction")
    print(f"    Explanation: {r.explanation}")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_opposite_directions()
    test_no_counter_flow()
    test_counter_flow_detected()
    test_at_threshold()
    test_below_threshold()
    test_all_stationary()
    test_empty_zone()
    test_custom_threshold()
    test_up_down_counter_flow()
    test_multiple_zones()
    test_explanation_content()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} ANOMALY TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
