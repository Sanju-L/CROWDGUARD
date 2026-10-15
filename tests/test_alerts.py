# ──────────────────────────────────────────────
# CrowdGuard -- Alert Management Unit Tests
# ──────────────────────────────────────────────

"""
Unit tests for Alert, AlertManager, all 6 alert types,
acknowledgements, cooldown deduplication, and CSV export.

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_alerts
"""

from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alerts.alert_manager import Alert, AlertManager, AlertSeverity, AlertType
from analytics.anomaly import CounterFlowReport
from analytics.bottleneck import GateReport
from analytics.density import ZoneReport
from analytics.flow import ZoneFlow

# ── Test Harness ─────────────────────────────

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


# ── Test Cases ───────────────────────────────

def test_alert_types_and_severities():
    divider("Alert Types and Severities Definition")
    check(AlertType.HIGH_DENSITY == "HIGH_DENSITY", "AlertType.HIGH_DENSITY defined")
    check(AlertType.COUNTER_FLOW == "COUNTER_FLOW", "AlertType.COUNTER_FLOW defined")
    check(AlertType.BOTTLENECK == "BOTTLENECK", "AlertType.BOTTLENECK defined")
    check(AlertType.FLOW_INSTABILITY == "FLOW_INSTABILITY", "AlertType.FLOW_INSTABILITY defined")
    check(AlertType.EXIT_CONGESTION == "EXIT_CONGESTION", "AlertType.EXIT_CONGESTION defined")
    check(AlertType.ROUTE_CHANGED == "ROUTE_CHANGED", "AlertType.ROUTE_CHANGED defined")
    check(len(AlertType.ALL) == 6, "All 6 Alert Types present")

    check(AlertSeverity.INFO == "INFO", "AlertSeverity.INFO defined")
    check(AlertSeverity.WARNING == "WARNING", "AlertSeverity.WARNING defined")
    check(AlertSeverity.HIGH == "HIGH", "AlertSeverity.HIGH defined")
    check(AlertSeverity.CRITICAL == "CRITICAL", "AlertSeverity.CRITICAL defined")


def test_high_density_alert():
    divider("High Density Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    # 1. Normal density -> no alert
    zr_safe = [ZoneReport("Zone A", 10, 50, 20.0, "SAFE", (0, 0, 100, 100))]
    alerts = mgr.process_analytics(zone_reports=zr_safe, now_ts=100.0)
    check(len(alerts) == 0, "SAFE density triggers no alerts")

    # 2. Critical density -> CRITICAL alert
    zr_crit = [ZoneReport("Zone B", 48, 50, 96.0, "CRITICAL", (0, 0, 100, 100))]
    alerts = mgr.process_analytics(zone_reports=zr_crit, now_ts=101.0)
    check(len(alerts) == 1, "CRITICAL density triggers 1 alert")
    check(alerts[0].alert_type == AlertType.HIGH_DENSITY, "Alert type is HIGH_DENSITY")
    check(alerts[0].severity == AlertSeverity.CRITICAL, "Severity is CRITICAL")
    check("CRITICAL density" in alerts[0].explanation, "Explanation details CRITICAL density")


def test_counter_flow_alert():
    divider("Counter-Flow Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    cf = CounterFlowReport(
        zone_name="Zone A",
        dominant_direction="RIGHT",
        opposite_direction="LEFT",
        dominant_count=6,
        opposite_count=4,
        moving_count=10,
        opposing_pct=40.0,
        counter_flow=True,
        threshold=30.0,
        explanation="40% opposing",
    )
    alerts = mgr.process_analytics(counter_flow_reports=[cf], now_ts=100.0)
    check(len(alerts) == 1, "Counter-flow triggers 1 alert")
    check(alerts[0].alert_type == AlertType.COUNTER_FLOW, "Alert type is COUNTER_FLOW")
    check(alerts[0].zone == "Zone A", "Zone is Zone A")
    check("Counter-flow detected in Zone A" in alerts[0].explanation, "Explanation mentions Zone A and opposing flow")


def test_bottleneck_alert():
    divider("Bottleneck Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    gr = GateReport(
        gate_name="Gate_AB",
        zone_a="Zone A",
        zone_b="Zone B",
        crossings_a_to_b=15,
        crossings_b_to_a=2,
        inflow_rate=30.0,
        outflow_rate=4.0,
        accumulation_rate=26.0,
        bottleneck=True,
        threshold=10.0,
        explanation="Bottleneck at Gate_AB",
    )
    alerts = mgr.process_analytics(gate_reports=[gr], now_ts=100.0)
    check(len(alerts) == 1, "Bottleneck triggers 1 alert")
    check(alerts[0].alert_type == AlertType.BOTTLENECK, "Alert type is BOTTLENECK")
    check(alerts[0].severity == AlertSeverity.CRITICAL, "Severe accumulation triggers CRITICAL")
    check("Gate_AB" in alerts[0].explanation, "Explanation contains gate name")


def test_flow_instability_alert():
    divider("Flow Instability Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    # 8 people stationary out of 10 -> 80% stagnant
    zf = ZoneFlow(
        zone_name="Zone C",
        dominant_direction="STATIONARY",
        average_speed=2.1,
        moving_count=2,
        stationary_count=8,
        total_count=10,
    )
    alerts = mgr.process_analytics(zone_flows=[zf], now_ts=100.0)
    check(len(alerts) == 1, "High stationary count triggers FLOW_INSTABILITY alert")
    check(alerts[0].alert_type == AlertType.FLOW_INSTABILITY, "Alert type is FLOW_INSTABILITY")
    check("Flow instability in Zone C" in alerts[0].explanation, "Explanation notes stagnant crowd")


def test_exit_congestion_alert():
    divider("Exit Congestion Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    zr_exit = [ZoneReport("EXIT_A", 45, 50, 90.0, "CRITICAL", (0, 0, 100, 100))]
    alerts = mgr.process_analytics(zone_reports=zr_exit, now_ts=100.0)
    exit_alts = [a for a in alerts if a.alert_type == AlertType.EXIT_CONGESTION]
    check(len(exit_alts) == 1, "Exit node at CRITICAL density triggers EXIT_CONGESTION")
    check(exit_alts[0].zone == "EXIT_A", "Zone is EXIT_A")


def test_route_changed_alert():
    divider("Route Changed Alert Triggering")
    mgr = AlertManager(cooldown_seconds=0.0)

    @dataclass
    class _MockRoute:
        recommended_exit: str
        route_nodes: list

    # Frame 1: Route to EXIT_A
    r1 = _MockRoute(recommended_exit="EXIT_A", route_nodes=["ZONE_A", "ZONE_C", "EXIT_A"])
    alts1 = mgr.process_analytics(route_result=r1, now_ts=100.0)
    check(len([a for a in alts1 if a.alert_type == AlertType.ROUTE_CHANGED]) == 0, "Initial route setup does not trigger change alert")

    # Frame 2: Route changes to EXIT_B
    r2 = _MockRoute(recommended_exit="EXIT_B", route_nodes=["ZONE_A", "ZONE_B", "EXIT_B"])
    alts2 = mgr.process_analytics(route_result=r2, now_ts=101.0)
    route_alts = [a for a in alts2 if a.alert_type == AlertType.ROUTE_CHANGED]
    check(len(route_alts) == 1, "Route switch triggers ROUTE_CHANGED alert")
    check("EXIT_B" in route_alts[0].explanation, "Explanation includes new exit")


def test_cooldown_and_acknowledgement():
    divider("Cooldown Deduplication & Acknowledgement")
    mgr = AlertManager(cooldown_seconds=5.0)

    zr = [ZoneReport("Zone A", 45, 50, 90.0, "HIGH", (0, 0, 100, 100))]

    # Time = 0s
    a1 = mgr.process_analytics(zone_reports=zr, now_ts=0.0)
    check(len(a1) == 1, "First event creates alert")

    # Time = 2s (within 5s cooldown) -> deduplicated
    a2 = mgr.process_analytics(zone_reports=zr, now_ts=2.0)
    check(len(a2) == 0, "Duplicate event within cooldown is deduplicated")

    # Time = 6s (after cooldown) -> allowed
    a3 = mgr.process_analytics(zone_reports=zr, now_ts=6.0)
    check(len(a3) == 1, "Event after cooldown window creates new alert")

    # Active vs Historical
    check(len(mgr.get_active_alerts()) == 2, "2 active alerts initially")
    check(len(mgr.get_historical_alerts()) == 2, "2 historical alerts logged")

    # Acknowledge single
    first_id = a1[0].id
    check(mgr.acknowledge(first_id) is True, "Single alert acknowledged successfully")
    check(len(mgr.get_active_alerts()) == 1, "1 active alert remains after single acknowledge")

    # Acknowledge all
    count = mgr.acknowledge_all()
    check(count == 1, "Acknowledge all acknowledged 1 remaining alert")
    check(len(mgr.get_active_alerts()) == 0, "0 active alerts after acknowledge_all")
    check(len(mgr.get_historical_alerts()) == 2, "Historical alerts preserved after acknowledge_all")


def test_csv_export():
    divider("CSV Export Formatting")
    mgr = AlertManager(cooldown_seconds=0.0)
    mgr.create_alert("Zone A", AlertType.HIGH_DENSITY, AlertSeverity.CRITICAL, "High density alert", 1)
    mgr.create_alert("Gate_AB", AlertType.BOTTLENECK, AlertSeverity.WARNING, "Gate bottleneck", 2)

    csv_text = mgr.export_csv()
    check(len(csv_text) > 0, "CSV string generated")

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    header = rows[0]
    check("Alert ID" in header, "Header contains Alert ID")
    check("Timestamp" in header, "Header contains Timestamp")
    check("Zone" in header, "Header contains Zone")
    check("Alert Type" in header, "Header contains Alert Type")
    check("Severity" in header, "Header contains Severity")
    check("Explanation" in header, "Header contains Explanation")
    check("Acknowledged" in header, "Header contains Acknowledged")

    check(len(rows) == 3, "Header + 2 data rows present")


def run_all():
    test_alert_types_and_severities()
    test_high_density_alert()
    test_counter_flow_alert()
    test_bottleneck_alert()
    test_flow_instability_alert()
    test_exit_congestion_alert()
    test_route_changed_alert()
    test_cooldown_and_acknowledgement()
    test_csv_export()

    divider("RESULTS")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print(f"  [OK] ALL {PASS} ALERT SYSTEM TESTS PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
