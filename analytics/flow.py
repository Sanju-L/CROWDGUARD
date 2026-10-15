# ──────────────────────────────────────────────
# CrowdGuard -- Crowd Flow Analyser
# ──────────────────────────────────────────────

"""
Computes per-person and per-zone movement flow from
tracked centroid histories.

Mathematics used (fully transparent):
    displacement  = pos[t] - pos[t-n]
    speed         = ||displacement|| / (n / fps)   [px/s]
    angle         = atan2(dy, dx)                  [degrees]
    direction     = classify angle into LEFT/RIGHT/UP/DOWN

No opaque AI scores -- every value is a direct geometric
calculation from centroid positions.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from config import DENSITY_ZONES


# ── Constants ────────────────────────────────

STATIONARY_SPEED_THRESHOLD: float = 5.0   # px/s below which a person is "stationary"
MIN_HISTORY_POINTS: int = 2               # need at least 2 points for displacement

DIRECTION_LABELS = ("RIGHT", "DOWN", "LEFT", "UP")  # mapped from angle quadrants


# ── Data Structures ──────────────────────────


@dataclass
class PersonFlow:
    """Flow vector for a single tracked person."""

    track_id: int
    dx: float                         # horizontal displacement (px)
    dy: float                         # vertical displacement (px)
    speed: float                      # px/s
    angle_deg: float                  # movement angle (0=right, 90=down, ...)
    direction: str                    # LEFT | RIGHT | UP | DOWN | STATIONARY
    center: Tuple[float, float] = (0.0, 0.0)   # current position


@dataclass
class ZoneFlow:
    """Aggregated flow statistics for one zone."""

    zone_name: str
    dominant_direction: str           # most common direction
    direction_distribution: Dict[str, int] = field(default_factory=dict)
    average_speed: float = 0.0        # px/s
    moving_count: int = 0
    stationary_count: int = 0
    total_count: int = 0


# ── Pure math helpers ────────────────────────


def compute_displacement(
    history: List[Tuple[float, float]],
    lookback: int = 5,
) -> Tuple[float, float]:
    """
    Displacement vector between the current position and
    a position ``lookback`` frames earlier.

    Returns (dx, dy) in pixels.  Positive dx = rightward,
    positive dy = downward (OpenCV convention).
    """
    if len(history) < MIN_HISTORY_POINTS:
        return 0.0, 0.0

    n = min(lookback, len(history) - 1)
    x_now, y_now = history[-1]
    x_old, y_old = history[-(n + 1)]
    return x_now - x_old, y_now - y_old


def compute_speed(dx: float, dy: float, frames: int, fps: float) -> float:
    """
    Speed in pixels per second.

        speed = sqrt(dx^2 + dy^2) / (frames / fps)
    """
    if frames <= 0 or fps <= 0:
        return 0.0
    distance = math.hypot(dx, dy)
    elapsed = frames / fps
    return distance / elapsed


def compute_angle(dx: float, dy: float) -> float:
    """
    Movement angle in degrees [0, 360).

        0   = right
        90  = down
        180 = left
        270 = up
    """
    return math.degrees(math.atan2(dy, dx)) % 360


def classify_direction(
    speed: float,
    angle_deg: float,
    stationary_threshold: float = STATIONARY_SPEED_THRESHOLD,
) -> str:
    """
    Map speed + angle to a cardinal direction label.

    If speed < threshold -> STATIONARY.
    Otherwise split the 360-degree circle into 4 quadrants:

        315-45   -> RIGHT
         45-135  -> DOWN
        135-225  -> LEFT
        225-315  -> UP
    """
    if speed < stationary_threshold:
        return "STATIONARY"

    # Normalise to [0, 360)
    a = angle_deg % 360

    if a < 45 or a >= 315:
        return "RIGHT"
    if 45 <= a < 135:
        return "DOWN"
    if 135 <= a < 225:
        return "LEFT"
    return "UP"


# ── Flow Analyser ────────────────────────────


class FlowAnalyser:
    """
    Computes per-person flow vectors and per-zone flow
    statistics from tracked centroid histories.

    Usage
    -----
    >>> analyser = FlowAnalyser(fps=15)
    >>> person_flows = analyser.compute_person_flows(histories)
    >>> zone_flows   = analyser.compute_zone_flows(person_flows, frame_shape)
    """

    def __init__(
        self,
        fps: float = 15.0,
        lookback: int = 5,
        stationary_threshold: float = STATIONARY_SPEED_THRESHOLD,
        zones: Dict[str, dict] | None = None,
    ) -> None:
        self.fps = fps
        self.lookback = lookback
        self.stationary_threshold = stationary_threshold
        self.zones = zones or DENSITY_ZONES

    # ── Per-person ───────────────────────────

    def compute_person_flows(
        self,
        histories: Dict[int, List[Tuple[float, float]]],
    ) -> List[PersonFlow]:
        """
        Compute flow vector for every tracked person.

        Parameters
        ----------
        histories : {track_id: [(cx, cy), ...]}
            Position history per track (oldest first).

        Returns
        -------
        list[PersonFlow]
        """
        flows: List[PersonFlow] = []

        for tid, hist in histories.items():
            hist_list = list(hist)  # handle deques transparently
            if len(hist_list) < MIN_HISTORY_POINTS:
                continue

            dx, dy = compute_displacement(hist_list, self.lookback)
            n = min(self.lookback, len(hist_list) - 1)
            speed = compute_speed(dx, dy, n, self.fps)
            angle = compute_angle(dx, dy)
            direction = classify_direction(speed, angle, self.stationary_threshold)

            flows.append(
                PersonFlow(
                    track_id=tid,
                    dx=dx,
                    dy=dy,
                    speed=round(speed, 2),
                    angle_deg=round(angle, 2),
                    direction=direction,
                    center=hist_list[-1],
                )
            )

        return flows

    # ── Per-zone ─────────────────────────────

    def compute_zone_flows(
        self,
        person_flows: List[PersonFlow],
        frame_shape: Tuple[int, int],
    ) -> List[ZoneFlow]:
        """
        Aggregate per-person flows into per-zone statistics.

        Parameters
        ----------
        person_flows : list[PersonFlow]
        frame_shape  : (height, width)

        Returns
        -------
        list[ZoneFlow]
        """
        h, w = frame_shape
        zone_flows: List[ZoneFlow] = []

        for zone_name, cfg in self.zones.items():
            fx1, fy1, fx2, fy2 = cfg["bounds"]
            px1, py1 = fx1 * w, fy1 * h
            px2, py2 = fx2 * w, fy2 * h

            # Collect flows whose centre falls inside this zone
            in_zone = [
                pf for pf in person_flows
                if px1 <= pf.center[0] < px2 and py1 <= pf.center[1] < py2
            ]

            if not in_zone:
                zone_flows.append(
                    ZoneFlow(
                        zone_name=zone_name,
                        dominant_direction="STATIONARY",
                        direction_distribution={},
                        average_speed=0.0,
                        moving_count=0,
                        stationary_count=0,
                        total_count=0,
                    )
                )
                continue

            # Direction distribution
            dir_counts: Dict[str, int] = Counter(pf.direction for pf in in_zone)

            # Dominant direction (most common, excluding STATIONARY if others exist)
            non_stationary = {k: v for k, v in dir_counts.items() if k != "STATIONARY"}
            if non_stationary:
                dominant = max(non_stationary, key=non_stationary.get)  # type: ignore[arg-type]
            else:
                dominant = "STATIONARY"

            moving = sum(1 for pf in in_zone if pf.direction != "STATIONARY")
            stationary = len(in_zone) - moving
            avg_speed = sum(pf.speed for pf in in_zone) / len(in_zone)

            zone_flows.append(
                ZoneFlow(
                    zone_name=zone_name,
                    dominant_direction=dominant,
                    direction_distribution=dict(dir_counts),
                    average_speed=round(avg_speed, 2),
                    moving_count=moving,
                    stationary_count=stationary,
                    total_count=len(in_zone),
                )
            )

        return zone_flows

    # ── Visualisation ────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        person_flows: List[PersonFlow],
        zone_flows: List[ZoneFlow],
        frame_shape: Tuple[int, int] | None = None,
        zones: Dict[str, dict] | None = None,
        *,
        arrow_color: Tuple[int, int, int] = (0, 255, 255),
        arrow_scale: float = 0.4,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw movement arrows and zone flow summaries.

        - Each moving person gets a yellow arrow from their centre
          in their movement direction.
        - Each zone gets a dominant-direction indicator and speed label.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        _zones = zones or DENSITY_ZONES

        # ── Per-person arrows ────────────────
        for pf in person_flows:
            if pf.direction == "STATIONARY":
                continue
            cx, cy = int(pf.center[0]), int(pf.center[1])
            # Scale displacement for visual clarity
            ex = int(cx + pf.dx * arrow_scale)
            ey = int(cy + pf.dy * arrow_scale)
            cv2.arrowedLine(
                annotated, (cx, cy), (ex, ey),
                arrow_color, 2, tipLength=0.35,
            )

        # ── Per-zone summary ─────────────────
        direction_arrows = {
            "RIGHT": "->", "LEFT": "<-",
            "UP": "^", "DOWN": "v",
            "STATIONARY": "o",
        }

        for zf in zone_flows:
            cfg = _zones.get(zf.zone_name)
            if cfg is None:
                continue

            fx1, fy1, fx2, fy2 = cfg["bounds"]
            px1 = int(fx1 * w)
            py1 = int(fy1 * h)
            px2 = int(fx2 * w)

            arrow_sym = direction_arrows.get(zf.dominant_direction, "?")
            line1 = f"Flow: {arrow_sym} {zf.dominant_direction}"
            line2 = f"Spd: {zf.average_speed:.0f} px/s | Mov: {zf.moving_count} Stn: {zf.stationary_count}"

            # Position: bottom area of each zone
            text_x = px1 + 8
            zone_h = int((fy2 - fy1) * h)
            text_y1 = py1 + zone_h - 40
            text_y2 = py1 + zone_h - 18

            # Background
            (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            max_tw = max(tw1, tw2)
            cv2.rectangle(
                annotated,
                (text_x - 4, text_y1 - th1 - 6),
                (text_x + max_tw + 8, text_y2 + 6),
                (0, 0, 0), cv2.FILLED,
            )

            cv2.putText(
                annotated, line1, (text_x, text_y1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, line2, (text_x, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

        return annotated
