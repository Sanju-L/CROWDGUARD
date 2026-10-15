# ──────────────────────────────────────────────
# CrowdGuard -- Explainable Risk Engine
# ──────────────────────────────────────────────

"""
Calculates a transparent, weighted crowd-risk score (0-100)
from normalised analytics inputs and generates a human-readable
explanation of every contributing factor.

Formula
-------
    risk_score = SUM( weight_i * normalised_score_i )

    where each normalised_score_i is in [0, 100] and
    SUM(weight_i) = 1.0.

Risk levels (PROTOTYPE scoring bands):
    0  - 39   SAFE
    40 - 59   WARNING
    60 - 79   HIGH
    80 - 100  CRITICAL

Normalisation rules (documented per factor):

    density          = max zone density_pct, capped at 100
    counter_flow     = max zone opposing_pct, capped at 100
    bottleneck       = max |accumulation_rate| / threshold * 100, capped at 100
    flow_instability = (stationary_pct across all zones), capped at 100
    exit_congestion  = avg density of exit zones, capped at 100

All values are derived from transparent geometric / counting
calculations.  No opaque ML scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import (
    RISK_ALERT_THRESHOLD,
    RISK_LEVELS,
    RISK_WEIGHTS,
)


# ── Data Structures ──────────────────────────


@dataclass
class RiskFactor:
    """One contributing factor to the overall risk score."""

    name: str
    raw_value: float          # original metric value
    normalised: float         # 0 - 100
    weight: float             # 0.0 - 1.0
    weighted_score: float     # normalised * weight
    description: str          # human-readable note


@dataclass
class RiskReport:
    """Complete explainable risk assessment."""

    risk_score: float                              # 0 - 100
    risk_level: str                                # SAFE | WARNING | HIGH | CRITICAL
    alert: bool                                    # True if score >= alert threshold
    contributing_factors: List[RiskFactor] = field(default_factory=list)
    explanation: str = ""


# ── Classification ───────────────────────────


def classify_risk(score: float, levels: Dict[str, Tuple[int, int]] | None = None) -> str:
    """
    Map a 0-100 score to a risk level label.

    NOTE: These are *prototype* scoring bands, not universal
    safety standards.
    """
    _levels = levels or RISK_LEVELS
    for label, (lo, hi) in _levels.items():
        if lo <= score <= hi:
            return label
    return "CRITICAL"  # anything > 100 or unexpected


# ── Normalisation helpers ────────────────────


def normalise_density(zone_reports: list) -> Tuple[float, str]:
    """
    Normalisation:
        Take the maximum density_pct across all zones.
        Cap at 100.

    Returns (normalised_value, description).
    """
    if not zone_reports:
        return 0.0, "No zone data available"

    max_density = max(r.density_pct for r in zone_reports)
    norm = min(max_density, 100.0)
    worst_zone = max(zone_reports, key=lambda r: r.density_pct)

    if norm >= 75:
        desc = f"High density in {worst_zone.name}: {max_density:.0f}% of capacity"
    elif norm >= 50:
        desc = f"Moderate density in {worst_zone.name}: {max_density:.0f}% of capacity"
    else:
        desc = f"Low density (max {max_density:.0f}%)"

    return round(norm, 2), desc


def normalise_counter_flow(counter_flow_reports: list) -> Tuple[float, str]:
    """
    Normalisation:
        Take the maximum opposing_pct across zones with counter-flow.
        Cap at 100.
    """
    if not counter_flow_reports:
        return 0.0, "No counter-flow data available"

    flagged = [r for r in counter_flow_reports if r.counter_flow]

    if not flagged:
        max_pct = max(r.opposing_pct for r in counter_flow_reports)
        return round(min(max_pct, 100.0), 2), f"No counter-flow detected (max opposing {max_pct:.0f}%)"

    worst = max(flagged, key=lambda r: r.opposing_pct)
    norm = min(worst.opposing_pct, 100.0)
    return round(norm, 2), f"Counter-flow detected in {worst.zone_name}: {worst.opposing_pct:.0f}% opposing"


def normalise_bottleneck(
    gate_reports: list,
    threshold: float = 10.0,
) -> Tuple[float, str]:
    """
    Normalisation:
        Take the maximum |accumulation_rate| across gates.
        Normalise as: |accum| / threshold * 100, capped at 100.

    A gate at exactly the threshold scores 100.
    """
    if not gate_reports:
        return 0.0, "No gate data available"

    max_accum = max(abs(r.accumulation_rate) for r in gate_reports)
    norm = min(max_accum / max(threshold, 1e-9) * 100, 100.0)

    flagged = [r for r in gate_reports if r.bottleneck]
    if flagged:
        worst = max(flagged, key=lambda r: abs(r.accumulation_rate))
        desc = f"Bottleneck at {worst.gate_name}: {worst.accumulation_rate:+.1f} p/min"
    else:
        desc = f"No bottleneck (max accumulation {max_accum:.1f} p/min)"

    return round(norm, 2), desc


def normalise_flow_instability(zone_flows: list) -> Tuple[float, str]:
    """
    Normalisation:
        Calculate the overall percentage of stationary people
        across all zones.  High stationary % in a crowd
        indicates stalled flow (potential crush risk).

        stationary_pct = total_stationary / total_count * 100
        Capped at 100.
    """
    if not zone_flows:
        return 0.0, "No flow data available"

    total = sum(zf.total_count for zf in zone_flows)
    stationary = sum(zf.stationary_count for zf in zone_flows)

    if total == 0:
        return 0.0, "No people tracked"

    pct = stationary / total * 100
    norm = min(pct, 100.0)

    if norm >= 60:
        desc = f"High flow instability: {pct:.0f}% of crowd is stationary"
    elif norm >= 30:
        desc = f"Moderate flow instability: {pct:.0f}% stationary"
    else:
        desc = f"Normal flow ({pct:.0f}% stationary)"

    return round(norm, 2), desc


def normalise_exit_congestion(
    zone_reports: list,
    exit_zones: List[str] | None = None,
) -> Tuple[float, str]:
    """
    Normalisation:
        Average density_pct of designated exit zones.
        If no exit zones are specified, use the bottom zones
        (Zone C, Zone D) as default exits.
        Capped at 100.
    """
    _exit_zones = exit_zones or ["Zone C", "Zone D"]

    if not zone_reports:
        return 0.0, "No zone data available"

    exits = [r for r in zone_reports if r.name in _exit_zones]
    if not exits:
        return 0.0, "No exit zone data"

    avg = sum(r.density_pct for r in exits) / len(exits)
    norm = min(avg, 100.0)

    if norm >= 70:
        desc = f"Exit congestion: average {avg:.0f}% at exits"
    elif norm >= 40:
        desc = f"Moderate exit load: {avg:.0f}%"
    else:
        desc = f"Exits clear ({avg:.0f}%)"

    return round(norm, 2), desc


# ── Risk Engine ──────────────────────────────


class RiskEngine:
    """
    Explainable weighted risk scorer.

    Accepts outputs from the analytics pipeline and produces
    a ``RiskReport`` with a score, level, contributing factors,
    and human-readable explanation.

    Usage
    -----
    >>> engine = RiskEngine()
    >>> report = engine.evaluate(
    ...     zone_reports=density_reports,
    ...     counter_flow_reports=cf_reports,
    ...     gate_reports=gate_reports,
    ...     zone_flows=zone_flows,
    ... )
    >>> print(report.risk_score, report.risk_level)
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        levels: Dict[str, Tuple[int, int]] | None = None,
        alert_threshold: float = RISK_ALERT_THRESHOLD,
        exit_zones: List[str] | None = None,
        bottleneck_threshold: float = 10.0,
    ) -> None:
        self.weights = weights or RISK_WEIGHTS
        self.levels = levels or RISK_LEVELS
        self.alert_threshold = alert_threshold
        self.exit_zones = exit_zones
        self.bottleneck_threshold = bottleneck_threshold

    def evaluate(
        self,
        zone_reports: list | None = None,
        counter_flow_reports: list | None = None,
        gate_reports: list | None = None,
        zone_flows: list | None = None,
    ) -> RiskReport:
        """
        Compute the overall risk score.

        Parameters
        ----------
        zone_reports : list[ZoneReport]
            From ``DensityAnalyser.compute()``.
        counter_flow_reports : list[CounterFlowReport]
            From ``CounterFlowDetector.analyse()``.
        gate_reports : list[GateReport]
            From ``BottleneckDetector.get_reports()``.
        zone_flows : list[ZoneFlow]
            From ``FlowAnalyser.compute_zone_flows()``.

        Returns
        -------
        RiskReport
        """
        factors: List[RiskFactor] = []

        # ── 1. Density ───────────────────────
        density_norm, density_desc = normalise_density(zone_reports or [])
        factors.append(self._make_factor(
            "density", density_norm, self.weights["density"], density_desc,
            raw=density_norm,
        ))

        # ── 2. Counter-flow ──────────────────
        cf_norm, cf_desc = normalise_counter_flow(counter_flow_reports or [])
        factors.append(self._make_factor(
            "counter_flow", cf_norm, self.weights["counter_flow"], cf_desc,
            raw=cf_norm,
        ))

        # ── 3. Bottleneck ────────────────────
        bn_norm, bn_desc = normalise_bottleneck(
            gate_reports or [], self.bottleneck_threshold,
        )
        factors.append(self._make_factor(
            "bottleneck", bn_norm, self.weights["bottleneck"], bn_desc,
            raw=bn_norm,
        ))

        # ── 4. Flow instability ──────────────
        fi_norm, fi_desc = normalise_flow_instability(zone_flows or [])
        factors.append(self._make_factor(
            "flow_instability", fi_norm, self.weights["flow_instability"], fi_desc,
            raw=fi_norm,
        ))

        # ── 5. Exit congestion ───────────────
        ec_norm, ec_desc = normalise_exit_congestion(
            zone_reports or [], self.exit_zones,
        )
        factors.append(self._make_factor(
            "exit_congestion", ec_norm, self.weights["exit_congestion"], ec_desc,
            raw=ec_norm,
        ))

        # ── Weighted sum ─────────────────────
        risk_score = round(sum(f.weighted_score for f in factors), 2)
        risk_score = min(max(risk_score, 0.0), 100.0)

        risk_level = classify_risk(risk_score, self.levels)
        alert = risk_score >= self.alert_threshold

        # ── Explanation ──────────────────────
        explanation = self._build_explanation(risk_score, risk_level, factors)

        return RiskReport(
            risk_score=risk_score,
            risk_level=risk_level,
            alert=alert,
            contributing_factors=factors,
            explanation=explanation,
        )

    @staticmethod
    def _make_factor(
        name: str,
        normalised: float,
        weight: float,
        description: str,
        raw: float = 0.0,
    ) -> RiskFactor:
        return RiskFactor(
            name=name,
            raw_value=raw,
            normalised=normalised,
            weight=weight,
            weighted_score=round(normalised * weight, 2),
            description=description,
        )

    @staticmethod
    def _build_explanation(
        score: float,
        level: str,
        factors: List[RiskFactor],
    ) -> str:
        lines = [
            f"Risk Score: {score:.0f}",
            f"Risk Level: {level}",
            "",
            "NOTE: These are prototype scoring bands, not universal safety standards.",
            "",
            "Contributing factors (weight x normalised score = weighted score):",
        ]

        # Sort by weighted_score descending
        sorted_factors = sorted(factors, key=lambda f: f.weighted_score, reverse=True)
        for f in sorted_factors:
            lines.append(
                f"  - {f.name} ({f.weight:.0%}): "
                f"{f.normalised:.0f}/100 -> {f.weighted_score:.1f} pts"
            )
            if f.description:
                lines.append(f"    {f.description}")

        # Highlight top reasons
        significant = [f for f in sorted_factors if f.weighted_score >= 5.0]
        if significant:
            lines.append("")
            lines.append("Reasons:")
            for f in significant:
                lines.append(f"  - {f.description}")

        return "\n".join(lines)
