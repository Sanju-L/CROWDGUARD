# ──────────────────────────────────────────────
# CrowdGuard – Alert Management System
# ──────────────────────────────────────────────

"""
Centralised Alert Management System for CrowdGuard.

Every detected crowd safety anomaly creates an explainable
Alert with timestamp, zone, alert type, severity, and explanation.

Supported Alert Types:
  - HIGH_DENSITY
  - COUNTER_FLOW
  - BOTTLENECK
  - FLOW_INSTABILITY
  - EXIT_CONGESTION
  - ROUTE_CHANGED

Supports:
  - In-memory alert logging for current operator session
  - Operator acknowledgement
  - Intelligent cooldown deduplication
  - CSV export for reporting and audit logs
"""

from __future__ import annotations

import csv
import io
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class AlertType:
    HIGH_DENSITY = "HIGH_DENSITY"
    COUNTER_FLOW = "COUNTER_FLOW"
    BOTTLENECK = "BOTTLENECK"
    FLOW_INSTABILITY = "FLOW_INSTABILITY"
    EXIT_CONGESTION = "EXIT_CONGESTION"
    ROUTE_CHANGED = "ROUTE_CHANGED"

    ALL = (
        HIGH_DENSITY,
        COUNTER_FLOW,
        BOTTLENECK,
        FLOW_INSTABILITY,
        EXIT_CONGESTION,
        ROUTE_CHANGED,
    )


class AlertSeverity:
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    ALL = (INFO, WARNING, HIGH, CRITICAL)


@dataclass
class Alert:
    """Individual crowd safety alert record."""

    id: str
    timestamp: str
    zone: str
    alert_type: str
    severity: str
    explanation: str
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    frame_number: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    """
    Session-based Alert Manager.

    Inspects analytics outputs on each frame and raises structured
    alerts with cooldowns to prevent alarm fatigue.
    """

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.alerts: List[Alert] = []
        self._last_alert_time: Dict[Tuple[str, str], float] = {}
        self._prev_route_exit: Optional[str] = None
        self._prev_route_nodes: Optional[List[str]] = None

    def create_alert(
        self,
        zone: str,
        alert_type: str,
        severity: str,
        explanation: str,
        frame_number: int = 0,
        now_ts: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Create and record an alert if not within the cooldown period
        for the same (zone, alert_type).
        """
        current_time = time.time() if now_ts is None else now_ts
        key = (zone, alert_type)

        if key in self._last_alert_time:
            last_time = self._last_alert_time[key]
            if (current_time - last_time) < self.cooldown_seconds:
                return None

        self._last_alert_time[key] = current_time

        alert_id = f"ALT-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
        iso_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert = Alert(
            id=alert_id,
            timestamp=iso_timestamp,
            zone=zone,
            alert_type=alert_type,
            severity=severity,
            explanation=explanation,
            acknowledged=False,
            frame_number=frame_number,
        )

        self.alerts.append(alert)
        return alert

    def process_analytics(
        self,
        zone_reports: list = None,
        counter_flow_reports: list = None,
        gate_reports: list = None,
        zone_flows: list = None,
        route_result: any = None,
        frame_number: int = 0,
        now_ts: Optional[float] = None,
    ) -> List[Alert]:
        """
        Scan all real-time analytics outputs for anomalies and trigger alerts.
        """
        new_alerts: List[Alert] = []

        # 1. High Density
        if zone_reports:
            for zr in zone_reports:
                if zr.status == "CRITICAL":
                    alt = self.create_alert(
                        zone=zr.name,
                        alert_type=AlertType.HIGH_DENSITY,
                        severity=AlertSeverity.CRITICAL,
                        explanation=f"{zr.name} reached CRITICAL density at {zr.density_pct:.0f}% of capacity ({zr.people_count}/{zr.capacity} people).",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)
                elif zr.status == "HIGH":
                    alt = self.create_alert(
                        zone=zr.name,
                        alert_type=AlertType.HIGH_DENSITY,
                        severity=AlertSeverity.HIGH,
                        explanation=f"{zr.name} at HIGH density ({zr.density_pct:.0f}% of capacity).",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)
                elif zr.status == "WARNING":
                    alt = self.create_alert(
                        zone=zr.name,
                        alert_type=AlertType.HIGH_DENSITY,
                        severity=AlertSeverity.WARNING,
                        explanation=f"{zr.name} density elevated to WARNING ({zr.density_pct:.0f}%).",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)

        # 2. Counter-Flow
        if counter_flow_reports:
            for cf in counter_flow_reports:
                if cf.counter_flow:
                    sev = AlertSeverity.CRITICAL if cf.opposing_pct >= 50.0 else AlertSeverity.WARNING
                    alt = self.create_alert(
                        zone=cf.zone_name,
                        alert_type=AlertType.COUNTER_FLOW,
                        severity=sev,
                        explanation=f"Counter-flow detected in {cf.zone_name}: {cf.opposing_pct:.0f}% ({cf.opposite_count}/{cf.moving_count}) moving {cf.opposite_direction}, opposing {cf.dominant_direction} flow.",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)

        # 3. Bottleneck
        if gate_reports:
            for gr in gate_reports:
                if gr.bottleneck:
                    sev = AlertSeverity.CRITICAL if abs(gr.accumulation_rate) >= 20.0 else AlertSeverity.WARNING
                    alt = self.create_alert(
                        zone=gr.gate_name,
                        alert_type=AlertType.BOTTLENECK,
                        severity=sev,
                        explanation=f"Bottleneck at {gr.gate_name}: Net accumulation rate {gr.accumulation_rate:+.1f} people/min between {gr.zone_a} and {gr.zone_b}.",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)

        # 4. Flow Instability / Stagnation
        if zone_flows:
            for zf in zone_flows:
                if zf.total_count >= 8:
                    stagnant_pct = (zf.stationary_count / zf.total_count) * 100.0
                    if stagnant_pct >= 70.0:
                        alt = self.create_alert(
                            zone=zf.zone_name,
                            alert_type=AlertType.FLOW_INSTABILITY,
                            severity=AlertSeverity.WARNING,
                            explanation=f"Flow instability in {zf.zone_name}: {stagnant_pct:.0f}% ({zf.stationary_count}/{zf.total_count}) people stationary. Avg speed {zf.average_speed:.1f} px/s.",
                            frame_number=frame_number,
                            now_ts=now_ts,
                        )
                        if alt:
                            new_alerts.append(alt)

        # 5. Exit Congestion
        if zone_reports:
            for zr in zone_reports:
                if "EXIT" in zr.name.upper() and zr.status in ("HIGH", "CRITICAL"):
                    alt = self.create_alert(
                        zone=zr.name,
                        alert_type=AlertType.EXIT_CONGESTION,
                        severity=AlertSeverity.CRITICAL if zr.status == "CRITICAL" else AlertSeverity.HIGH,
                        explanation=f"Exit congestion at {zr.name}: {zr.density_pct:.0f}% capacity reached.",
                        frame_number=frame_number,
                        now_ts=now_ts,
                    )
                    if alt:
                        new_alerts.append(alt)

        # 6. Route Changed
        if route_result:
            current_exit = getattr(route_result, "recommended_exit", None)
            current_nodes = getattr(route_result, "route_nodes", None)

            if (
                self._prev_route_exit is not None
                and current_exit is not None
                and (current_exit != self._prev_route_exit or current_nodes != self._prev_route_nodes)
            ):
                prev_exit = self._prev_route_exit
                alt = self.create_alert(
                    zone=current_exit,
                    alert_type=AlertType.ROUTE_CHANGED,
                    severity=AlertSeverity.HIGH,
                    explanation=f"Evacuation route changed from {prev_exit} to {current_exit}. New path: {' -> '.join(current_nodes)}.",
                    frame_number=frame_number,
                    now_ts=now_ts,
                )
                if alt:
                    new_alerts.append(alt)

            if current_exit:
                self._prev_route_exit = current_exit
            if current_nodes:
                self._prev_route_nodes = list(current_nodes)

        return new_alerts

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an individual alert as acknowledged."""
        for a in self.alerts:
            if a.id == alert_id:
                a.acknowledged = True
                a.acknowledged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return True
        return False

    def acknowledge_all(self) -> int:
        """Mark all unacknowledged alerts as acknowledged."""
        count = 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for a in self.alerts:
            if not a.acknowledged:
                a.acknowledged = True
                a.acknowledged_at = now
                count += 1
        return count

    def get_active_alerts(self) -> List[Alert]:
        """Return all active (unacknowledged) alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def get_historical_alerts(self) -> List[Alert]:
        """Return all alerts logged during this session."""
        return list(self.alerts)

    def export_csv(self) -> str:
        """Export all session alerts to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Alert ID",
            "Timestamp",
            "Zone",
            "Alert Type",
            "Severity",
            "Explanation",
            "Acknowledged",
            "Acknowledged At",
            "Frame Number",
        ])

        for a in self.alerts:
            writer.writerow([
                a.id,
                a.timestamp,
                a.zone,
                a.alert_type,
                a.severity,
                a.explanation,
                "YES" if a.acknowledged else "NO",
                a.acknowledged_at or "",
                a.frame_number,
            ])

        return output.getvalue()

    def clear(self) -> None:
        """Reset session alerts."""
        self.alerts.clear()
        self._last_alert_time.clear()
        self._prev_route_exit = None
        self._prev_route_nodes = None
