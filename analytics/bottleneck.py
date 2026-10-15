# ──────────────────────────────────────────────
# CrowdGuard -- Bottleneck Detector
# ──────────────────────────────────────────────

"""
Detects bottlenecks at configurable virtual gates between
zones by tracking zone transitions of individual persons
over a rolling time window.

Formulas (fully documented):

    For each gate connecting Zone-A <-> Zone-B:

    crossing_a_to_b : person was in Zone-A, now in Zone-B
    crossing_b_to_a : person was in Zone-B, now in Zone-A

    Over a rolling window of W seconds:

        crossings_a_to_b = count of a->b crossings in window
        crossings_b_to_a = count of b->a crossings in window

        inflow_rate   = crossings_a_to_b / W * 60     [people/min]
        outflow_rate  = crossings_b_to_a / W * 60     [people/min]

        accumulation_rate = inflow_rate - outflow_rate [people/min]
        (positive = people accumulating on the zone_b side)

    If |accumulation_rate| >= threshold  ->  BOTTLENECK

Every value is a direct count / time ratio.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from config import (
    BOTTLENECK_ACCUMULATION_THRESHOLD,
    BOTTLENECK_GATES,
    BOTTLENECK_WINDOW_SECONDS,
    DENSITY_ZONES,
)


# ── Data Structures ──────────────────────────


@dataclass
class GateReport:
    """Bottleneck assessment for one virtual gate."""

    gate_name: str
    zone_a: str
    zone_b: str
    crossings_a_to_b: int              # raw count in window
    crossings_b_to_a: int              # raw count in window
    inflow_rate: float                 # people/min  (a -> b)
    outflow_rate: float                # people/min  (b -> a)
    accumulation_rate: float           # people/min  (net toward b)
    bottleneck: bool
    threshold: float
    explanation: str


# ── Zone assignment helper ───────────────────


def assign_zone(
    cx: float,
    cy: float,
    frame_shape: Tuple[int, int],
    zones: Dict[str, dict] | None = None,
) -> str | None:
    """
    Return the zone name that contains centroid (cx, cy),
    or None if outside all zones.
    """
    _zones = zones or DENSITY_ZONES
    h, w = frame_shape

    for name, cfg in _zones.items():
        fx1, fy1, fx2, fy2 = cfg["bounds"]
        px1, py1 = fx1 * w, fy1 * h
        px2, py2 = fx2 * w, fy2 * h
        if px1 <= cx < px2 and py1 <= cy < py2:
            return name
    return None


# ── Bottleneck Detector ──────────────────────


class BottleneckDetector:
    """
    Tracks zone transitions at virtual gates and flags
    bottlenecks when net accumulation exceeds a threshold.

    Usage
    -----
    >>> detector = BottleneckDetector(fps=15)
    >>> # Each frame:
    >>> detector.update(track_zones, frame_number)
    >>> reports = detector.get_reports()
    """

    def __init__(
        self,
        fps: float = 15.0,
        window_seconds: float = BOTTLENECK_WINDOW_SECONDS,
        threshold: float = BOTTLENECK_ACCUMULATION_THRESHOLD,
        gates: Dict[str, dict] | None = None,
        zones: Dict[str, dict] | None = None,
    ) -> None:
        self.fps = fps
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.gates = gates or BOTTLENECK_GATES
        self.zones = zones or DENSITY_ZONES

        # Rolling crossing logs:  deque of (frame_number, direction)
        # direction: "a_to_b" or "b_to_a"
        self._crossings: Dict[str, Deque[Tuple[int, str]]] = {
            gname: deque() for gname in self.gates
        }

        # Previous frame zone assignment per track_id
        self._prev_zones: Dict[int, str | None] = {}

    # ── Per-frame update ─────────────────────

    def update(
        self,
        track_zones: Dict[int, str | None],
        frame_number: int,
    ) -> None:
        """
        Update crossing logs with zone transitions.

        Parameters
        ----------
        track_zones : {track_id: zone_name_or_None}
            Current zone assignment for every active track.
        frame_number : int
            Sequential frame index.
        """
        window_frames = int(self.window_seconds * self.fps)

        for tid, cur_zone in track_zones.items():
            prev_zone = self._prev_zones.get(tid)

            # Detect transition
            if prev_zone is not None and cur_zone is not None and prev_zone != cur_zone:
                self._record_crossing(prev_zone, cur_zone, frame_number)

        # Prune old crossings outside the rolling window
        cutoff = frame_number - window_frames
        for gname in self._crossings:
            while self._crossings[gname] and self._crossings[gname][0][0] < cutoff:
                self._crossings[gname].popleft()

        # Store current zones for next frame
        self._prev_zones = dict(track_zones)

    def _record_crossing(
        self, from_zone: str, to_zone: str, frame_number: int
    ) -> None:
        """Record a zone transition as a gate crossing."""
        for gname, gcfg in self.gates.items():
            za, zb = gcfg["zone_a"], gcfg["zone_b"]

            if from_zone == za and to_zone == zb:
                self._crossings[gname].append((frame_number, "a_to_b"))
            elif from_zone == zb and to_zone == za:
                self._crossings[gname].append((frame_number, "b_to_a"))

    # ── Report generation ────────────────────

    def get_reports(self) -> List[GateReport]:
        """
        Compute bottleneck reports for all gates based on
        current crossing logs.

        Returns
        -------
        list[GateReport]

        Formulas
        --------
        inflow_rate       = crossings_a_to_b / window_seconds * 60
        outflow_rate      = crossings_b_to_a / window_seconds * 60
        accumulation_rate = inflow_rate - outflow_rate
        bottleneck        = |accumulation_rate| >= threshold
        """
        reports: List[GateReport] = []

        for gname, gcfg in self.gates.items():
            crossings = self._crossings[gname]

            a_to_b = sum(1 for _, d in crossings if d == "a_to_b")
            b_to_a = sum(1 for _, d in crossings if d == "b_to_a")

            # Rates in people/min
            #   count / window_seconds  -> people/second
            #   * 60                    -> people/minute
            w = max(self.window_seconds, 1e-9)
            inflow_rate = round(a_to_b / w * 60, 2)
            outflow_rate = round(b_to_a / w * 60, 2)
            accumulation_rate = round(inflow_rate - outflow_rate, 2)

            bottleneck = abs(accumulation_rate) >= self.threshold

            explanation = self._build_explanation(
                gname, gcfg, inflow_rate, outflow_rate,
                accumulation_rate, a_to_b, b_to_a, bottleneck,
            )

            reports.append(
                GateReport(
                    gate_name=gname,
                    zone_a=gcfg["zone_a"],
                    zone_b=gcfg["zone_b"],
                    crossings_a_to_b=a_to_b,
                    crossings_b_to_a=b_to_a,
                    inflow_rate=inflow_rate,
                    outflow_rate=outflow_rate,
                    accumulation_rate=accumulation_rate,
                    bottleneck=bottleneck,
                    threshold=self.threshold,
                    explanation=explanation,
                )
            )

        return reports

    def _build_explanation(
        self,
        gate_name: str,
        gcfg: dict,
        inflow: float,
        outflow: float,
        accumulation: float,
        a_to_b: int,
        b_to_a: int,
        bottleneck: bool,
    ) -> str:
        za, zb = gcfg["zone_a"], gcfg["zone_b"]

        if bottleneck:
            if accumulation > 0:
                side = zb
            else:
                side = za
            return (
                f"{gate_name}: BOTTLENECK detected. "
                f"Inflow {inflow:.1f} p/min ({za}->{zb}), "
                f"Outflow {outflow:.1f} p/min ({zb}->{za}), "
                f"Net accumulation {accumulation:+.1f} p/min "
                f"toward {side}. "
                f"Exceeds {self.threshold:.0f} p/min threshold."
            )

        return (
            f"{gate_name}: Normal flow. "
            f"Inflow {inflow:.1f} p/min, Outflow {outflow:.1f} p/min, "
            f"Net {accumulation:+.1f} p/min. "
            f"Below {self.threshold:.0f} p/min threshold."
        )

    # ── Helpers ──────────────────────────────

    def reset(self) -> None:
        """Clear all state."""
        for gname in self._crossings:
            self._crossings[gname].clear()
        self._prev_zones.clear()

    # ── Visualisation ────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        reports: List[GateReport],
        gates: Dict[str, dict] | None = None,
        *,
        font_scale: float = 0.45,
    ) -> np.ndarray:
        """
        Draw virtual gate lines and bottleneck indicators.

        - Normal gates: thin cyan line
        - Bottleneck gates: thick red line + warning label
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        _gates = gates or BOTTLENECK_GATES

        for rpt in reports:
            gcfg = _gates.get(rpt.gate_name)
            if gcfg is None:
                continue

            fx1, fy1, fx2, fy2 = gcfg["line"]
            px1, py1 = int(fx1 * w), int(fy1 * h)
            px2, py2 = int(fx2 * w), int(fy2 * h)

            if rpt.bottleneck:
                color = (0, 0, 255)       # red
                thickness = 3
            else:
                color = (255, 255, 0)     # cyan
                thickness = 1

            # Gate line
            cv2.line(annotated, (px1, py1), (px2, py2), color, thickness)

            # Label at midpoint
            mx = (px1 + px2) // 2
            my = (py1 + py2) // 2

            label = f"{rpt.gate_name}: {rpt.accumulation_rate:+.0f} p/min"
            (tw, th_t), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                annotated,
                (mx - tw // 2 - 3, my - th_t - 5),
                (mx + tw // 2 + 3, my + 5),
                (0, 0, 0), cv2.FILLED,
            )
            cv2.putText(
                annotated, label,
                (mx - tw // 2, my),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, 1, cv2.LINE_AA,
            )

            # Bottleneck warning badge
            if rpt.bottleneck:
                badge = "BOTTLENECK"
                (bw, bh), _ = cv2.getTextSize(
                    badge, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )
                cv2.rectangle(
                    annotated,
                    (mx - bw // 2 - 3, my + 8),
                    (mx + bw // 2 + 3, my + bh + 16),
                    (0, 0, 255), cv2.FILLED,
                )
                cv2.putText(
                    annotated, badge,
                    (mx - bw // 2, my + bh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

        return annotated
