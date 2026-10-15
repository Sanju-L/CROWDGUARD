"""Temporal, geometry-based localization of gate-like openings in video."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass
class DetectedGate:
    id: str
    corners: tuple[tuple[float, float], ...]
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    confidence: float
    orientation_degrees: float
    frames_seen: int = 1
    missed_scans: int = 0
    occluded: bool = False
    people_count_nearby: int = 0


class GateDetector:
    """Find persistent doorway-shaped openings and keep stable IDs across frames.

    Gates are inferred from paired vertical jambs and a connecting lintel. Person
    movement is not required as evidence, so quiet but visible gates can be found.
    This deliberately does not classify or label objects inside the opening.
    """

    SCAN_INTERVAL = 10
    MIN_OBSERVATIONS = 2
    MAX_MISSED_SCANS = 8

    def __init__(self) -> None:
        self.gates: list[DetectedGate] = []
        self.next_id = 1
        self.frame_count = 0
        self._frame_shape: tuple[int, int] | None = None

    def update(self, frame: np.ndarray, tracks: Any = ()) -> None:
        """Analyze a frame and update persistent gate locations."""
        if frame is None or frame.ndim < 2 or frame.size == 0:
            raise ValueError("Gate detection requires a non-empty image frame")

        height, width = frame.shape[:2]
        self._frame_shape = (height, width)
        self.frame_count += 1

        if self.frame_count == 1 or self.frame_count % self.SCAN_INTERVAL == 0:
            self._update_candidates(self._find_candidates(frame), width, height)

        for gate in self.gates:
            gate.people_count_nearby = sum(
                1
                for track in tracks
                if math.dist(
                    (float(track.center_x), float(track.center_y)),
                    gate.center,
                )
                < max(80, min(width, height) * 0.18)
            )

    @staticmethod
    def _find_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
        """Extract doorway-like quadrilaterals from stable, strong line evidence."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 130)

        min_vertical = max(36, int(height * 0.16))
        min_horizontal = max(28, int(width * 0.045))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(24, min_vertical // 2),
            minLineLength=min(min_vertical, min_horizontal),
            maxLineGap=max(8, int(min(width, height) * 0.025)),
        )
        if lines is None:
            return []

        verticals: list[tuple[float, float, float, float]] = []
        horizontals: list[tuple[float, float, float, float]] = []
        for raw_line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(float, raw_line)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            if abs(dx) / length <= 0.16 and length >= min_vertical:
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                verticals.append((x1, y1, x2, y2))
            elif abs(dy) / length <= 0.16 and length >= min_horizontal:
                if x1 > x2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                horizontals.append((x1, y1, x2, y2))

        candidates: list[dict[str, Any]] = []
        for index, first in enumerate(verticals):
            for second in verticals[index + 1 :]:
                left, right = sorted(
                    (first, second), key=lambda line: (line[0] + line[2]) / 2
                )
                rx_top, ry_top, rx_bottom, ry_bottom = right
                lx_top, ly_top, lx_bottom, ly_bottom = left
                left_height = ly_bottom - ly_top
                right_height = ry_bottom - ry_top
                if min(left_height, right_height) < min_vertical:
                    continue

                left_x = (lx_top + lx_bottom) / 2
                right_x = (rx_top + rx_bottom) / 2

                gate_width = right_x - left_x
                top_y = (ly_top + ry_top) / 2
                bottom_y = min(ly_bottom, ry_bottom)
                gate_height = bottom_y - top_y
                if gate_width < width * 0.045 or gate_width > width * 0.48:
                    continue
                if gate_height < height * 0.30:
                    continue
                aspect = gate_width / gate_height
                if not 0.18 <= aspect <= 0.9:
                    continue
                if abs(ly_top - ry_top) > height * 0.07:
                    continue
                if abs(left_height - right_height) > max(height * 0.12, gate_height * 0.3):
                    continue

                lintel = GateDetector._find_lintel(
                    horizontals, left_x, right_x, top_y, width, height
                )
                if lintel is None:
                    continue
                if GateDetector._has_multiple_crossbars(
                    horizontals, left_x, right_x, top_y, bottom_y, height
                ):
                    continue
                hx1, hy1, hx2, hy2 = lintel
                lintel_angle = math.degrees(math.atan2(hy2 - hy1, hx2 - hx1))

                corners = (
                    (lx_top, ly_top),
                    (rx_top, ry_top),
                    (rx_bottom, ry_bottom),
                    (lx_bottom, ly_bottom),
                )
                x1 = max(0, int(min(point[0] for point in corners)))
                y1 = max(0, int(min(point[1] for point in corners)))
                x2 = min(width, int(max(point[0] for point in corners)))
                y2 = min(height, int(max(point[1] for point in corners)))
                if x2 <= x1 or y2 <= y1:
                    continue

                top_alignment = max(0.0, 1 - abs(ly_top - ry_top) / max(height * 0.07, 1))
                height_alignment = max(
                    0.0,
                    1 - abs(left_height - right_height) / max(height * 0.12, gate_height * 0.3),
                )
                confidence = min(0.88, 0.56 + 0.14 * top_alignment + 0.12 * height_alignment)
                candidates.append({
                    "corners": corners,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "confidence": confidence,
                    "orientation_degrees": lintel_angle,
                })

        return sorted(
            GateDetector._suppress_duplicates(candidates),
            key=lambda candidate: (candidate["center"][0], candidate["center"][1]),
        )

    @staticmethod
    def _find_lintel(
        horizontals: list[tuple[float, float, float, float]],
        left_x: float,
        right_x: float,
        top_y: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float] | None:
        tolerance_x = max(frame_width * 0.025, (right_x - left_x) * 0.16)
        tolerance_y = frame_height * 0.045
        segments = []
        for line in horizontals:
            x1, y1, x2, y2 = line
            if abs((y1 + y2) / 2 - top_y) <= tolerance_y:
                segments.append(line)
        if not segments:
            return None

        segments.sort(key=lambda line: line[0])
        max_gap = max(6, (right_x - left_x) * 0.12)
        merged: list[list[float]] = []
        for x1, y1, x2, y2 in segments:
            if merged and x1 <= merged[-1][2] + max_gap:
                merged[-1][2] = max(merged[-1][2], x2)
                center_y = (y1 + y2) / 2
                merged[-1][1] = merged[-1][3] = (merged[-1][1] + center_y) / 2
            else:
                merged.append([x1, (y1 + y2) / 2, x2, (y1 + y2) / 2])

        matches = [
            line for line in merged
            if line[0] <= left_x + tolerance_x and line[2] >= right_x - tolerance_x
        ]
        if not matches:
            return None
        return tuple(max(matches, key=lambda line: line[2] - line[0]))

    @staticmethod
    def _has_multiple_crossbars(
        horizontals: list[tuple[float, float, float, float]],
        left_x: float,
        right_x: float,
        top_y: float,
        bottom_y: float,
        frame_height: int,
    ) -> bool:
        width = right_x - left_x
        interior_ys = sorted(
            (y1 + y2) / 2
            for x1, y1, x2, y2 in horizontals
            if top_y + (bottom_y - top_y) * 0.14 < (y1 + y2) / 2 < bottom_y - (bottom_y - top_y) * 0.14
            and x1 <= left_x + width * 0.2
            and x2 >= right_x - width * 0.2
        )
        distinct_rails = []
        for y in interior_ys:
            if not distinct_rails or y - distinct_rails[-1] > frame_height * 0.035:
                distinct_rails.append(y)
        return len(distinct_rails) >= 2

    @staticmethod
    def _suppress_duplicates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates, key=lambda item: item["confidence"], reverse=True
        ):
            if all(
                GateDetector._box_iou(candidate["bbox"], other["bbox"]) < 0.55
                for other in selected
            ):
                selected.append(candidate)
        return selected

    @staticmethod
    def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _update_candidates(
        self, candidates: list[dict[str, Any]], frame_width: int, frame_height: int
    ) -> None:
        unmatched = set(range(len(self.gates)))
        for candidate in candidates:
            best_index = None
            best_score = 0.0
            for index in unmatched:
                gate = self.gates[index]
                overlap = self._box_iou(candidate["bbox"], gate.bbox)
                distance = math.dist(candidate["center"], gate.center)
                proximity = max(0.0, 1 - distance / max(50, min(frame_width, frame_height) * 0.15))
                score = max(overlap, proximity * 0.7)
                if score > best_score:
                    best_index, best_score = index, score

            if best_index is not None and best_score >= 0.3:
                gate = self.gates[best_index]
                unmatched.remove(best_index)
                alpha = 0.35
                gate.corners = tuple(
                    (
                        old[0] * (1 - alpha) + new[0] * alpha,
                        old[1] * (1 - alpha) + new[1] * alpha,
                    )
                    for old, new in zip(gate.corners, candidate["corners"])
                )
                gate.bbox = tuple(
                    round(old * (1 - alpha) + new * alpha)
                    for old, new in zip(gate.bbox, candidate["bbox"])
                )
                gate.center = (
                    gate.center[0] * (1 - alpha) + candidate["center"][0] * alpha,
                    gate.center[1] * (1 - alpha) + candidate["center"][1] * alpha,
                )
                gate.orientation_degrees = (
                    gate.orientation_degrees * (1 - alpha)
                    + candidate["orientation_degrees"] * alpha
                )
                gate.frames_seen += 1
                gate.missed_scans = 0
                gate.occluded = False
                gate.confidence = min(
                    0.97, max(gate.confidence, candidate["confidence"]) + 0.015
                )
            else:
                x1, y1, x2, y2 = candidate["bbox"]
                self.gates.append(DetectedGate(
                    id=f"Gate {self.next_id}",
                    corners=candidate["corners"],
                    bbox=(x1, y1, x2, y2),
                    center=candidate["center"],
                    confidence=candidate["confidence"],
                    orientation_degrees=candidate["orientation_degrees"],
                ))
                self.next_id += 1

        for index in unmatched:
            gate = self.gates[index]
            gate.missed_scans += 1
            gate.occluded = gate.frames_seen >= self.MIN_OBSERVATIONS
            gate.confidence = max(0.0, gate.confidence - 0.04)
        self.gates = [
            gate for gate in self.gates
            if gate.missed_scans <= self.MAX_MISSED_SCANS and gate.confidence >= 0.2
        ]
        self._normalize_ids()

    def _normalize_ids(self) -> None:
        """Keep displayed gate labels contiguous and stable in image order.

        Candidate tracks can expire after missed scans, while newly detected
        candidates may be created later. Renumbering the surviving spatial
        tracks prevents labels such as Gate 6 from appearing when only two
        openings are visible.
        """
        self.gates.sort(key=lambda gate: (gate.center[0], gate.center[1]))
        for index, gate in enumerate(self.gates, start=1):
            gate.id = f"Gate {index}"
        self.next_id = len(self.gates) + 1

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay confirmed gate boundaries without altering the input frame."""
        annotated = frame.copy()
        visible_gates = [
            gate for gate in self.gates
            if gate.frames_seen >= self.MIN_OBSERVATIONS
        ]
        visible_gates.sort(key=lambda gate: (gate.center[0], gate.center[1]))
        for index, gate in enumerate(visible_gates, start=1):
            color = (0, 140, 255) if gate.occluded or gate.confidence < 0.65 else (0, 230, 118)
            points = np.round(gate.corners).astype(np.int32)
            cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
            top_center = (
                int(round((gate.corners[0][0] + gate.corners[1][0]) / 2)),
                int(round((gate.corners[0][1] + gate.corners[1][1]) / 2)),
            )
            bottom_center = (
                int(round((gate.corners[2][0] + gate.corners[3][0]) / 2)),
                int(round((gate.corners[2][1] + gate.corners[3][1]) / 2)),
            )
            cv2.arrowedLine(
                annotated,
                top_center,
                bottom_center,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            label = f"Gate {index} {gate.confidence:.0%}"
            if gate.occluded:
                label += " LOW CONFIDENCE"
            x1, y1, _, _ = gate.bbox
            label_y = max(18, y1 - 8)
            cv2.putText(
                annotated, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return annotated

    def to_api_dict(self, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        """Return confirmed gates in normalized camera-image coordinates."""
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("Frame dimensions must be positive")

        result = []
        visible_gates = [
            gate for gate in self.gates
            if gate.frames_seen >= self.MIN_OBSERVATIONS and gate.confidence >= 0.35
        ]
        visible_gates.sort(key=lambda gate: (gate.center[0], gate.center[1]))
        for index, gate in enumerate(visible_gates, start=1):
            x1, y1, x2, y2 = gate.bbox
            result.append({
                "id": f"Gate {index}",
                "name": f"Gate {index}",
                "bounds": [x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h],
                "corners": [
                    [
                        round(float(np.clip(x / frame_w, 0, 1)), 6),
                        round(float(np.clip(y / frame_h, 0, 1)), 6),
                    ]
                    for x, y in gate.corners
                ],
                "cx": gate.center[0] / frame_w,
                "cy": gate.center[1] / frame_h,
                "orientation_degrees": round(gate.orientation_degrees, 2),
                "coordinate_space": "camera-image-normalized",
                "confidence_basis": "structural_geometry_and_temporal_persistence",
                "status": "LOW_CONFIDENCE" if gate.occluded else "GEOMETRY_MATCH",
                "confidence": round(gate.confidence, 2),
                "occluded": gate.occluded,
                "frames_seen": gate.frames_seen,
                "people_nearby": gate.people_count_nearby,
            })
        return result
