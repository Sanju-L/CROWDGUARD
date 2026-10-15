# ──────────────────────────────────────────────
# CrowdGuard – Zone-Based Density Analyser
# ──────────────────────────────────────────────

"""
Divides the video frame into configurable rectangular zones
and computes per-zone crowd density from tracked person
centroids.

Each zone reports:
  - people_count   (int)
  - density_pct    (float, 0–100+)
  - status         (SAFE | NORMAL | WARNING | HIGH | CRITICAL)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import (
    DENSITY_STATUS_COLORS,
    DENSITY_THRESHOLDS,
    DENSITY_ZONES,
)


# ── Data Structures ──────────────────────────


@dataclass
class ZoneReport:
    """Density report for a single zone."""

    name: str
    people_count: int
    capacity: int
    density_pct: float          # 0.0 – 100.0+
    status: str                 # SAFE | NORMAL | WARNING | HIGH | CRITICAL
    bounds_px: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels


# ── Classification ───────────────────────────


def classify_density(
    density_fraction: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    """
    Classify a density fraction (people_count / capacity)
    into a human-readable status level.

    Parameters
    ----------
    density_fraction : float
        Ratio of current count to zone capacity (can exceed 1.0).
    thresholds : dict, optional
        Override the default thresholds from config.

    Returns
    -------
    str
        One of: SAFE, NORMAL, WARNING, HIGH, CRITICAL
    """
    th = thresholds or DENSITY_THRESHOLDS

    if density_fraction < th["SAFE"]:
        return "SAFE"
    if density_fraction < th["NORMAL"]:
        return "NORMAL"
    if density_fraction < th["WARNING"]:
        return "WARNING"
    if density_fraction < th["HIGH"]:
        return "HIGH"
    return "CRITICAL"


# ── Analyser ─────────────────────────────────


class DensityAnalyser:
    """
    Zone-based crowd-density estimator.

    Accepts a list of person centroids (from the tracker)
    and returns a ``ZoneReport`` for every configured zone.

    Usage
    -----
    >>> analyser = DensityAnalyser()
    >>> reports  = analyser.compute(centroids, frame_shape=(1080, 1920))
    >>> for r in reports:
    ...     print(r.name, r.people_count, r.status)
    """

    def __init__(
        self,
        zones: Dict[str, dict] | None = None,
        thresholds: Dict[str, float] | None = None,
    ) -> None:
        self.zones = zones or DENSITY_ZONES
        self.thresholds = thresholds or DENSITY_THRESHOLDS

    # ── Core computation ─────────────────────

    def compute(
        self,
        centroids: List[Tuple[float, float]],
        frame_shape: Tuple[int, int],
    ) -> List[ZoneReport]:
        """
        Count people per zone and classify density.

        Parameters
        ----------
        centroids : list of (cx, cy)
            Centre coordinates of each tracked person (pixels).
        frame_shape : (height, width)
            Dimensions of the current frame.

        Returns
        -------
        list[ZoneReport]
        """
        h, w = frame_shape
        reports: List[ZoneReport] = []

        for name, cfg in self.zones.items():
            fx1, fy1, fx2, fy2 = cfg["bounds"]
            capacity: int = cfg["capacity"]

            # Convert fractional → pixel bounds
            px1, py1 = int(fx1 * w), int(fy1 * h)
            px2, py2 = int(fx2 * w), int(fy2 * h)

            # Count centroids inside this zone
            count = 0
            for cx, cy in centroids:
                if px1 <= cx < px2 and py1 <= cy < py2:
                    count += 1

            # Density as a fraction of capacity
            density_frac = count / max(capacity, 1)
            density_pct = density_frac * 100.0

            status = classify_density(density_frac, self.thresholds)

            reports.append(
                ZoneReport(
                    name=name,
                    people_count=count,
                    capacity=capacity,
                    density_pct=round(density_pct, 1),
                    status=status,
                    bounds_px=(px1, py1, px2, py2),
                )
            )

        return reports

    # ── Visualisation ────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        reports: List[ZoneReport],
        *,
        alpha: float = 0.25,
        font_scale: float = 0.55,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw semi-transparent zone overlays with density labels.

        Each zone displays:
            Zone A: 15 people | 32%

        Returns
        -------
        np.ndarray
            Annotated BGR image (original is untouched).
        """
        overlay = frame.copy()
        annotated = frame.copy()

        for rpt in reports:
            x1, y1, x2, y2 = rpt.bounds_px
            color = DENSITY_STATUS_COLORS.get(rpt.status, (180, 180, 180))

            # Semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)

            # Zone border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Blend overlay onto annotated frame
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Draw text labels on top
        for rpt in reports:
            x1, y1, x2, y2 = rpt.bounds_px
            color = DENSITY_STATUS_COLORS.get(rpt.status, (180, 180, 180))

            label = f"{rpt.name}: {rpt.people_count} people | {rpt.density_pct:.0f}%"
            status_label = f"[{rpt.status}]"

            # Primary label (top of zone)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            text_y = y1 + th + 10
            cv2.rectangle(
                annotated,
                (x1 + 4, text_y - th - 4),
                (x1 + tw + 12, text_y + baseline + 2),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

            # Status badge below
            (sw, sh), sb = cv2.getTextSize(
                status_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            status_y = text_y + sh + 12
            cv2.rectangle(
                annotated,
                (x1 + 4, status_y - sh - 4),
                (x1 + sw + 12, status_y + sb + 2),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                status_label,
                (x1 + 8, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated
