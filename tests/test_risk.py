# ──────────────────────────────────────────────
# CrowdGuard -- Risk Engine Unit Tests
# ──────────────────────────────────────────────

"""
Tests using synthetic analytics data with hand-verifiable
weighted score calculations.

Formula verified:
    risk_score = SUM(weight_i * normalised_i)
    classify per prototype bands: SAFE/WARNING/HIGH/CRITICAL

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_risk
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass

from risk.risk_engine import (
    RiskEngine,
    RiskReport,
    classify_risk,
    normalise_bottleneck,
    normalise_counter_flow,
    normalise_density,
    normalise_exit_congestion,
    normalise_flow_instability,
)

# ── Mock data classes ────────────────────────
# Lightweight stand-ins so we don't import every analytics module.


@dataclass
class MockZoneReport:
    name: str
    density_pct: float
    people_count: int = 0
    capacity: int = 50
    status: str = "SAFE"
    bounds_px: tuple = (0, 0, 0, 0)


@dataclass
class MockCounterFlowReport:
    zone_name: str
    opposing_pct: float
    counter_flow: bool
    dominant_direction: str = "RIGHT"
    opposite_direction: str = "LEFT"
    dominant_count: int = 0
    opposite_count: int = 0
    moving_count: int = 10
    threshold: float = 30.0
    explanation: str = ""


@dataclass
class MockGateReport:
    gate_name: str
    accumulation_rate: float
    bottleneck: bool
    zone_a: str = "A"
    zone_b: str = "B"
    crossings_a_to_b: int = 0
    crossings_b_to_a: int = 0
    inflow_rate: float = 0.0
    outflow_rate: float = 0.0
    threshold: float = 10.0
    explanation: str = ""


@dataclass
class MockZoneFlow:
    zone_name: str
    total_count: int
    stationary_count: int
    moving_count: int = 0
    dominant_direction: str = "RIGHT"
    direction_distribution: dict = None
    average_speed: float = 50.0

    def __post_init__(self):
        if self.direction_distribution is None:
            self.direction_distribution = {}
        self.moving_count = self.total_count - self.stationary_count


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


def approx(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) < tol


def divider(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ── Test classify_risk ───────────────────────

def test_classify_risk() -> None:
    divider("classify_risk()")

    check(classify_risk(0) == "SAFE",       "0 -> SAFE")
    check(classify_risk(20) == "SAFE",      "20 -> SAFE")
    check(classify_risk(39) == "SAFE",      "39 -> SAFE")
    check(classify_risk(40) == "WARNING",   "40 -> WARNING")
    check(classify_risk(59) == "WARNING",   "59 -> WARNING")
    check(classify_risk(60) == "HIGH",      "60 -> HIGH")
    check(classify_risk(79) == "HIGH",      "79 -> HIGH")
    check(classify_risk(80) == "CRITICAL",  "80 -> CRITICAL")
    check(classify_risk(100) == "CRITICAL", "100 -> CRITICAL")


# ── Test normalise_density ───────────────────

def test_normalise_density() -> None:
    divider("normalise_density()")

    zones = [
        MockZoneReport("Zone A", density_pct=30.0),
        MockZoneReport("Zone B", density_pct=80.0),
    ]
    norm, desc = normalise_density(zones)
    check(approx(norm, 80.0), f"max density = 80 (got {norm})")
    check("Zone B" in desc, f"mentions worst zone")

    # Over 100 -> capped
    zones2 = [MockZoneReport("Zone A", density_pct=150.0)]
    norm2, _ = normalise_density(zones2)
    check(norm2 == 100.0, f"capped at 100 (got {norm2})")

    # Empty
    norm3, _ = normalise_density([])
    check(norm3 == 0.0, "empty -> 0")


# ── Test normalise_counter_flow ──────────────

def test_normalise_counter_flow() -> None:
    divider("normalise_counter_flow()")

    reports = [
        MockCounterFlowReport("Zone A", opposing_pct=15.0, counter_flow=False),
        MockCounterFlowReport("Zone B", opposing_pct=55.0, counter_flow=True),
    ]
    norm, desc = normalise_counter_flow(reports)
    check(approx(norm, 55.0), f"max opposing = 55 (got {norm})")
    check("Zone B" in desc, "mentions flagged zone")

    # No counter-flow flagged
    reports2 = [MockCounterFlowReport("Zone A", opposing_pct=10.0, counter_flow=False)]
    norm2, desc2 = normalise_counter_flow(reports2)
    check(approx(norm2, 10.0), f"unflagged max = 10 (got {norm2})")
    check("No counter-flow" in desc2, "says no counter-flow")

    # Empty
    norm3, _ = normalise_counter_flow([])
    check(norm3 == 0.0, "empty -> 0")


# ── Test normalise_bottleneck ────────────────

def test_normalise_bottleneck() -> None:
    divider("normalise_bottleneck()")

    # |accum| = 15, threshold = 10 -> 15/10*100 = 150 -> capped at 100
    reports = [MockGateReport("Gate_AB", accumulation_rate=15.0, bottleneck=True)]
    norm, desc = normalise_bottleneck(reports, threshold=10.0)
    check(norm == 100.0, f"15/10*100 capped at 100 (got {norm})")
    check("Bottleneck" in desc, "mentions bottleneck")

    # |accum| = 5, threshold = 10 -> 50
    reports2 = [MockGateReport("Gate_AB", accumulation_rate=5.0, bottleneck=False)]
    norm2, _ = normalise_bottleneck(reports2, threshold=10.0)
    check(approx(norm2, 50.0), f"5/10*100 = 50 (got {norm2})")

    # Negative accumulation
    reports3 = [MockGateReport("Gate_AB", accumulation_rate=-8.0, bottleneck=False)]
    norm3, _ = normalise_bottleneck(reports3, threshold=10.0)
    check(approx(norm3, 80.0), f"|-8|/10*100 = 80 (got {norm3})")

    # Empty
    norm4, _ = normalise_bottleneck([])
    check(norm4 == 0.0, "empty -> 0")


# ── Test normalise_flow_instability ──────────

def test_normalise_flow_instability() -> None:
    divider("normalise_flow_instability()")

    flows = [
        MockZoneFlow("Zone A", total_count=20, stationary_count=10),
        MockZoneFlow("Zone B", total_count=10, stationary_count=5),
    ]
    # total = 30, stationary = 15 -> 50%
    norm, desc = normalise_flow_instability(flows)
    check(approx(norm, 50.0), f"15/30*100 = 50 (got {norm})")

    # All moving
    flows2 = [MockZoneFlow("Zone A", total_count=20, stationary_count=0)]
    norm2, _ = normalise_flow_instability(flows2)
    check(norm2 == 0.0, f"0 stationary -> 0 (got {norm2})")

    # Empty
    norm3, _ = normalise_flow_instability([])
    check(norm3 == 0.0, "empty -> 0")


# ── Test normalise_exit_congestion ───────────

def test_normalise_exit_congestion() -> None:
    divider("normalise_exit_congestion()")

    zones = [
        MockZoneReport("Zone A", density_pct=20.0),
        MockZoneReport("Zone B", density_pct=30.0),
        MockZoneReport("Zone C", density_pct=80.0),
        MockZoneReport("Zone D", density_pct=60.0),
    ]
    # Default exits = Zone C, Zone D -> avg = (80+60)/2 = 70
    norm, desc = normalise_exit_congestion(zones)
    check(approx(norm, 70.0), f"exit avg = 70 (got {norm})")

    # Custom exits
    norm2, _ = normalise_exit_congestion(zones, exit_zones=["Zone A"])
    check(approx(norm2, 20.0), f"custom exit avg = 20 (got {norm2})")


# ── Test full evaluation: all safe ───────────

def test_all_safe() -> None:
    divider("Full evaluation: all inputs safe")

    engine = RiskEngine()
    report = engine.evaluate(
        zone_reports=[
            MockZoneReport("Zone A", density_pct=10.0),
            MockZoneReport("Zone C", density_pct=5.0),
            MockZoneReport("Zone D", density_pct=8.0),
        ],
        counter_flow_reports=[
            MockCounterFlowReport("Zone A", opposing_pct=5.0, counter_flow=False),
        ],
        gate_reports=[
            MockGateReport("Gate_AB", accumulation_rate=1.0, bottleneck=False),
        ],
        zone_flows=[
            MockZoneFlow("Zone A", total_count=10, stationary_count=1),
        ],
    )

    check(report.risk_score < 40, f"score < 40 (got {report.risk_score})")
    check(report.risk_level == "SAFE", f"level = SAFE (got {report.risk_level})")
    check(report.alert is False, "no alert")
    check(len(report.contributing_factors) == 5, "5 factors")
    print(f"    Score: {report.risk_score}, Level: {report.risk_level}")


# ── Test full evaluation: critical ───────────

def test_critical() -> None:
    divider("Full evaluation: critical scenario")

    engine = RiskEngine()
    report = engine.evaluate(
        zone_reports=[
            MockZoneReport("Zone A", density_pct=95.0),
            MockZoneReport("Zone B", density_pct=88.0),
            MockZoneReport("Zone C", density_pct=92.0),
            MockZoneReport("Zone D", density_pct=85.0),
        ],
        counter_flow_reports=[
            MockCounterFlowReport("Zone A", opposing_pct=60.0, counter_flow=True),
        ],
        gate_reports=[
            MockGateReport("Gate_AB", accumulation_rate=25.0, bottleneck=True),
        ],
        zone_flows=[
            MockZoneFlow("Zone A", total_count=40, stationary_count=30),
        ],
    )

    # Manual calculation:
    #   density:          min(95, 100) = 95 * 0.35 = 33.25
    #   counter_flow:     60 * 0.20 = 12.0
    #   bottleneck:       min(25/10*100, 100) = 100 * 0.20 = 20.0
    #   flow_instability: 30/40*100 = 75 * 0.15 = 11.25
    #   exit_congestion:  (92+85)/2 = 88.5 * 0.10 = 8.85
    #   total = 33.25 + 12.0 + 20.0 + 11.25 + 8.85 = 85.35
    expected = 33.25 + 12.0 + 20.0 + 11.25 + 8.85

    check(approx(report.risk_score, expected, tol=1.5), f"score ~{expected:.1f} (got {report.risk_score})")
    check(report.risk_level == "CRITICAL", f"level = CRITICAL (got {report.risk_level})")
    check(report.alert is True, "alert triggered")
    check("Risk Score" in report.explanation, "explanation has score")
    check("CRITICAL" in report.explanation, "explanation has level")
    check("prototype" in report.explanation.lower(), "explanation mentions prototype")
    print(f"    Score: {report.risk_score}, Level: {report.risk_level}")


# ── Test weighted sum correctness ────────────

def test_weighted_sum() -> None:
    divider("Weighted sum hand-verification")

    # All inputs at exactly 50/100
    engine = RiskEngine()
    report = engine.evaluate(
        zone_reports=[
            MockZoneReport("Zone A", density_pct=50.0),
            MockZoneReport("Zone C", density_pct=50.0),
            MockZoneReport("Zone D", density_pct=50.0),
        ],
        counter_flow_reports=[
            MockCounterFlowReport("Zone A", opposing_pct=50.0, counter_flow=True),
        ],
        gate_reports=[
            MockGateReport("Gate_AB", accumulation_rate=5.0, bottleneck=False),
        ],
        zone_flows=[
            MockZoneFlow("Zone A", total_count=10, stationary_count=5),
        ],
    )

    # density=50*0.35=17.5, cf=50*0.20=10, bn=50*0.20=10,
    # fi=50*0.15=7.5, ec=50*0.10=5  total=50
    check(approx(report.risk_score, 50.0, tol=2.0), f"uniform 50 inputs -> ~50 (got {report.risk_score})")
    check(report.risk_level == "WARNING", f"WARNING (got {report.risk_level})")


# ── Test custom weights ──────────────────────

def test_custom_weights() -> None:
    divider("Custom weights")

    custom_weights = {
        "density": 1.0,
        "counter_flow": 0.0,
        "bottleneck": 0.0,
        "flow_instability": 0.0,
        "exit_congestion": 0.0,
    }
    engine = RiskEngine(weights=custom_weights)
    report = engine.evaluate(
        zone_reports=[MockZoneReport("Zone A", density_pct=70.0)],
    )

    # Only density matters: 70 * 1.0 = 70
    check(approx(report.risk_score, 70.0), f"density-only = 70 (got {report.risk_score})")


# ── Test explanation content ─────────────────

def test_explanation_content() -> None:
    divider("Explanation content")

    engine = RiskEngine()
    report = engine.evaluate(
        zone_reports=[
            MockZoneReport("Zone A", density_pct=85.0),
            MockZoneReport("Zone C", density_pct=20.0),
            MockZoneReport("Zone D", density_pct=20.0),
        ],
        counter_flow_reports=[
            MockCounterFlowReport("Zone A", opposing_pct=45.0, counter_flow=True),
        ],
    )

    check("Risk Score" in report.explanation, "has 'Risk Score'")
    check("Risk Level" in report.explanation, "has 'Risk Level'")
    check("prototype" in report.explanation.lower(), "has 'prototype' disclaimer")
    check("Reasons:" in report.explanation, "has reasons section")
    print(f"\n{report.explanation}")


# ── Test empty inputs ────────────────────────

def test_empty_inputs() -> None:
    divider("Empty inputs (all None)")

    engine = RiskEngine()
    report = engine.evaluate()

    check(report.risk_score == 0.0, f"empty -> 0 (got {report.risk_score})")
    check(report.risk_level == "SAFE", f"SAFE (got {report.risk_level})")
    check(report.alert is False, "no alert")


# ── Run all ──────────────────────────────────

def main() -> None:
    test_classify_risk()
    test_normalise_density()
    test_normalise_counter_flow()
    test_normalise_bottleneck()
    test_normalise_flow_instability()
    test_normalise_exit_congestion()
    test_all_safe()
    test_critical()
    test_weighted_sum()
    test_custom_weights()
    test_explanation_content()
    test_empty_inputs()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} RISK ENGINE TESTS PASSED")
    else:
        print(f"  [!!] {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
