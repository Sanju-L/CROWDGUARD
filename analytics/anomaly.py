# ──────────────────────────────────────────────
# CrowdGuard -- Counter-Flow / Anomaly Detector
# ──────────────────────────────────────────────

"""
Rule-based, explainable anomaly detection focused on
counter-flow: people moving against the dominant crowd
direction inside a zone.

Algorithm (no ML, fully transparent):

    1. Read the directional distribution from flow analysis.
    2. Identify the dominant (most common) movement direction.
    3. Look up the opposite direction:
         LEFT  <-> RIGHT
         UP    <-> DOWN
    4. Calculate opposing_pct = opposite_count / moving_count * 100
    5. If opposing_pct >= threshold  ->  flag COUNTER_FLOW.

Every decision is traceable to a simple ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from analytics.flow import ZoneFlow


# ── Constants ────────────────────────────────

DEFAULT_COUNTER_FLOW_THRESHOLD: float = 30.0   # percent

OPPOSITE_DIRECTION: Dict[str, str] = {
    "LEFT":  "RIGHT",
    "RIGHT": "LEFT",
    "UP":    "DOWN",
    "DOWN":  "UP",
}


# ── Data Structures ──────────────────────────


@dataclass
class CounterFlowReport:
    """Counter-flow assessment for a single zone."""

    zone_name: str
    dominant_direction: str
    opposite_direction: str
    dominant_count: int
    opposite_count: int
    moving_count: int
    opposing_pct: float               # 0.0 - 100.0
    counter_flow: bool                # True if threshold exceeded
    threshold: float                  # the threshold used
    explanation: str                  # human-readable summary


# ── Detector ─────────────────────────────────


class CounterFlowDetector:
    """
    Rule-based counter-flow detector.

    Consumes ``ZoneFlow`` objects (from ``FlowAnalyser``)
    and flags zones where a significant fraction of people
    move against the dominant direction.

    Usage
    -----
    >>> detector = CounterFlowDetector(threshold=30.0)
    >>> reports  = detector.analyse(zone_flows)
    >>> for r in reports:
    ...     print(r.zone_name, r.counter_flow, r.explanation)
    """

    def __init__(self, threshold: float = DEFAULT_COUNTER_FLOW_THRESHOLD) -> None:
        self.threshold = threshold

    # ── Core analysis ────────────────────────

    def analyse(self, zone_flows: List[ZoneFlow]) -> List[CounterFlowReport]:
        """
        Evaluate counter-flow for every zone.

        Parameters
        ----------
        zone_flows : list[ZoneFlow]
            Output of ``FlowAnalyser.compute_zone_flows()``.

        Returns
        -------
        list[CounterFlowReport]
        """
        reports: List[CounterFlowReport] = []

        for zf in zone_flows:
            report = self._evaluate_zone(zf)
            reports.append(report)

        return reports

    def _evaluate_zone(self, zf: ZoneFlow) -> CounterFlowReport:
        """Evaluate a single zone for counter-flow."""

        dominant = zf.dominant_direction
        dist = zf.direction_distribution

        # Determine opposite direction
        opposite = OPPOSITE_DIRECTION.get(dominant, "")

        # Count people in each relevant direction
        dominant_count = dist.get(dominant, 0)
        opposite_count = dist.get(opposite, 0) if opposite else 0
        moving_count = zf.moving_count

        # Calculate opposing percentage (of MOVING people only)
        if moving_count > 0 and opposite:
            opposing_pct = round(opposite_count / moving_count * 100, 1)
        else:
            opposing_pct = 0.0

        # Flag counter-flow if threshold exceeded
        counter_flow = opposing_pct >= self.threshold

        # Build human-readable explanation
        explanation = self._build_explanation(
            zone_name=zf.zone_name,
            dominant=dominant,
            opposite=opposite,
            opposing_pct=opposing_pct,
            opposite_count=opposite_count,
            moving_count=moving_count,
            counter_flow=counter_flow,
        )

        return CounterFlowReport(
            zone_name=zf.zone_name,
            dominant_direction=dominant,
            opposite_direction=opposite,
            dominant_count=dominant_count,
            opposite_count=opposite_count,
            moving_count=moving_count,
            opposing_pct=opposing_pct,
            counter_flow=counter_flow,
            threshold=self.threshold,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        zone_name: str,
        dominant: str,
        opposite: str,
        opposing_pct: float,
        opposite_count: int,
        moving_count: int,
        counter_flow: bool,
    ) -> str:
        """Generate a human-readable, fully traceable explanation."""

        if dominant == "STATIONARY":
            return (
                f"{zone_name}: No dominant movement direction. "
                f"All tracked people are stationary."
            )

        if moving_count == 0:
            return f"{zone_name}: No moving people detected."

        if counter_flow:
            return (
                f"{zone_name}: Counter-flow detected: "
                f"{opposing_pct:.0f}% of tracked people "
                f"({opposite_count}/{moving_count}) are moving {opposite}, "
                f"opposite to the dominant direction ({dominant})."
            )

        return (
            f"{zone_name}: Normal flow. "
            f"{opposing_pct:.0f}% of tracked people move against "
            f"the dominant direction ({dominant}). "
            f"Below the {self.threshold:.0f}% threshold."
        )
