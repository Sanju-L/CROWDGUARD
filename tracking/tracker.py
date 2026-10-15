# ──────────────────────────────────────────────
# CrowdGuard – Multi-Object Tracker (ByteTrack & Crowd Tracking)
# ──────────────────────────────────────────────

"""
Wraps YOLO + ByteTrack / Spatial Association for multi-object
tracking with persistent IDs across consecutive frames.

Each tracked person carries an ID, bounding box, centre
coordinates, confidence, frame number, and a rolling
history of recent positions for downstream flow analysis.

Supports adaptive multi-tile detection for dense, high-resolution
overhead crowd footage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    BASE_DIR,
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    DETECTION_IOU_THRESHOLD,
    PERSON_CLASS_ID,
    YOLO_MODEL_NAME,
)
from detection.detector import nms_boxes


# ── Configuration ────────────────────────────

BYTETRACK_CFG = str(BASE_DIR / "bytetrack.yaml")
HISTORY_MAXLEN: int = 60          # keep last N positions per track


# ── Data Structures ──────────────────────────


@dataclass
class Track:
    """Snapshot of a single tracked person for the current frame."""

    track_id: int
    bbox: np.ndarray                           # [x1, y1, x2, y2]
    center_x: float
    center_y: float
    confidence: float
    frame_number: int
    class_name: str = "person"


@dataclass
class TrackerState:
    """
    Aggregated output from a single tracker update.

    Attributes
    ----------
    tracks : list[Track]
        Active tracks for the current frame.
    history : dict[int, deque[tuple[float, float]]]
        Rolling centroid history keyed by track_id.
    active_ids : set[int]
        Currently visible track IDs.
    total_unique_ids : int
        Cumulative unique IDs seen since reset.
    """

    tracks: List[Track] = field(default_factory=list)
    history: Dict[int, Deque[Tuple[float, float]]] = field(default_factory=dict)
    active_ids: set = field(default_factory=set)
    total_unique_ids: int = 0


# ── Tracker ──────────────────────────────────


class MultiObjectTracker:
    """
    Multi-object tracker using adaptive tiled YOLO detection
    and spatial trajectory association for robust crowd tracking.

    Usage
    -----
    >>> tracker = MultiObjectTracker()
    >>> state = tracker.update(frame, frame_number=0)
    >>> for t in state.tracks:
    ...     print(t.track_id, t.center_x, t.center_y)
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        tracker_cfg: str = BYTETRACK_CFG,
        history_maxlen: int = HISTORY_MAXLEN,
        classes: List[int] | None = None,
        max_disappeared: int = 15,
        max_distance: float = 70.0,
        model: YOLO | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.tracker_cfg = tracker_cfg
        self.history_maxlen = history_maxlen
        self.classes = classes if classes is not None else [0, 25]
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self._model: YOLO | None = model
        self._history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_maxlen)
        )
        self._seen_ids: set = set()

        # Spatial tracking state
        self._next_id: int = 1
        self._active_tracks: Dict[int, dict] = {}

    # ── Model lifecycle ──────────────────────

    def load_model(self) -> None:
        """Load the YOLO model (called lazily on first update)."""
        self._model = YOLO(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Core update ──────────────────────────

    def update(self, frame: np.ndarray, frame_number: int = 0) -> TrackerState:
        """
        Process one frame through adaptive YOLO crowd detector + tracker.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.
        frame_number : int
            Sequential frame index (for timestamps).

        Returns
        -------
        TrackerState
            Current tracks, history, and ID stats.
        """
        if not self.is_loaded:
            self.load_model()

        h, w = frame.shape[:2]

        # 1. Detect candidate people/crowd objects
        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        if False: # TILING DISABLED FOR EXTREME SPEED
            grid_r = 3 if h >= 1000 else 2
            grid_c = 2 if w >= 700 else 1

            crop_h = h // grid_r
            crop_w = w // grid_c
            overlap = 40

            for r in range(grid_r):
                for c in range(grid_c):
                    y1 = max(0, r * crop_h - overlap)
                    y2 = min(h, (r + 1) * crop_h + overlap)
                    x1 = max(0, c * crop_w - overlap)
                    x2 = min(w, (c + 1) * crop_w + overlap)

                    crop = frame[y1:y2, x1:x2]
                    res = self._model(
                        crop,
                        imgsz=320, # Reduced size for speed
                        conf=self.confidence,
                        classes=self.classes,
                        verbose=False,
                    )
                    if res and res[0].boxes is not None:
                        for box in res[0].boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            all_boxes.append([x1 + bx1, y1 + by1, x1 + bx2, y1 + by2])
                            all_scores.append(conf)
        else:
            results = self._model(
                frame,
                imgsz=DETECTION_IMAGE_SIZE,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.classes,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)

        # 2. Filter via NMS
        detections: List[dict] = []
        if all_boxes:
            boxes_arr = np.array(all_boxes, dtype=np.float32)
            scores_arr = np.array(all_scores, dtype=np.float32)
            keep = nms_boxes(boxes_arr, scores_arr, self.iou_threshold)
            for i in keep:
                b = boxes_arr[i]
                cx = float((b[0] + b[2]) / 2.0)
                cy = float((b[1] + b[3]) / 2.0)
                detections.append({
                    "bbox": b,
                    "center": (cx, cy),
                    "conf": float(scores_arr[i]),
                })

        # 3. Associate with active tracks
        current_tracks: List[Track] = []
        active_ids: set = set()

        if len(self._active_tracks) == 0:
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._active_tracks[tid] = {
                    "bbox": det["bbox"],
                    "center": det["center"],
                    "disappeared": 0,
                    "conf": det["conf"],
                }
                self._history[tid].append(det["center"])
                self._seen_ids.add(tid)
                active_ids.add(tid)
                current_tracks.append(
                    Track(
                        track_id=tid,
                        bbox=det["bbox"],
                        center_x=det["center"][0],
                        center_y=det["center"][1],
                        confidence=det["conf"],
                        frame_number=frame_number,
                    )
                )
        else:
            track_ids = list(self._active_tracks.keys())
            prev_centers = [self._active_tracks[tid]["center"] for tid in track_ids]

            if len(detections) > 0:
                det_centers = [det["center"] for det in detections]
                D = np.linalg.norm(
                    np.array(prev_centers)[:, None, :] - np.array(det_centers)[None, :, :],
                    axis=2,
                )

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for r, c in zip(rows, cols):
                    if r in used_rows or c in used_cols:
                        continue
                    if D[r, c] > self.max_distance:
                        continue

                    tid = track_ids[r]
                    det = detections[c]
                    self._active_tracks[tid]["bbox"] = det["bbox"]
                    self._active_tracks[tid]["center"] = det["center"]
                    self._active_tracks[tid]["conf"] = det["conf"]
                    self._active_tracks[tid]["disappeared"] = 0
                    self._history[tid].append(det["center"])
                    active_ids.add(tid)

                    current_tracks.append(
                        Track(
                            track_id=tid,
                            bbox=det["bbox"],
                            center_x=det["center"][0],
                            center_y=det["center"][1],
                            confidence=det["conf"],
                            frame_number=frame_number,
                        )
                    )

                    used_rows.add(r)
                    used_cols.add(c)

                # Unmatched tracks
                unused_rows = set(range(len(track_ids))) - used_rows
                for r in unused_rows:
                    tid = track_ids[r]
                    self._active_tracks[tid]["disappeared"] += 1
                    if self._active_tracks[tid]["disappeared"] > self.max_disappeared:
                        del self._active_tracks[tid]

                # Unmatched detections -> new tracks
                unused_cols = set(range(len(detections))) - used_cols
                for c in unused_cols:
                    det = detections[c]
                    tid = self._next_id
                    self._next_id += 1
                    self._active_tracks[tid] = {
                        "bbox": det["bbox"],
                        "center": det["center"],
                        "disappeared": 0,
                        "conf": det["conf"],
                    }
                    self._history[tid].append(det["center"])
                    self._seen_ids.add(tid)
                    active_ids.add(tid)

                    current_tracks.append(
                        Track(
                            track_id=tid,
                            bbox=det["bbox"],
                            center_x=det["center"][0],
                            center_y=det["center"][1],
                            confidence=det["conf"],
                            frame_number=frame_number,
                        )
                    )
            else:
                for tid in list(self._active_tracks.keys()):
                    self._active_tracks[tid]["disappeared"] += 1
                    if self._active_tracks[tid]["disappeared"] > self.max_disappeared:
                        del self._active_tracks[tid]

        return TrackerState(
            tracks=current_tracks,
            history=dict(self._history),
            active_ids=active_ids,
            total_unique_ids=len(self._seen_ids),
        )

    # ── Convenience ──────────────────────────

    def get_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Return the position history for a single track."""
        return list(self._history.get(track_id, []))

    def reset(self) -> None:
        """Clear all state (tracks, history, seen IDs)."""
        self._history.clear()
        self._seen_ids.clear()
        self._active_tracks.clear()
        self._next_id = 1
        self._model = None

    # ── Visualisation ────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        tracks: List[Track],
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw bounding boxes with ID labels on a copy of frame.
        """
        annotated = frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = t.bbox.astype(int)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Centre dot
            cv2.circle(
                annotated,
                (int(t.center_x), int(t.center_y)),
                4,
                (0, 0, 255),
                cv2.FILLED,
            )

            # Compact ID label
            label = f"ID {t.track_id}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated
