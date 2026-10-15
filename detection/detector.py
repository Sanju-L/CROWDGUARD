# ──────────────────────────────────────────────
# CrowdGuard – Person Detector
# ──────────────────────────────────────────────

"""
YOLO-based person and crowd detector using Ultralytics.
Accepts an OpenCV BGR frame and returns structured
Detection objects with bounding boxes, confidence,
centre coordinates, and class name.

Supports adaptive multi-tile scanning for high-density,
elevated, and high-resolution crowd surveillance video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DETECTION_CONFIDENCE,
    DETECTION_IOU_THRESHOLD,
    PERSON_CLASS_ID,
    YOLO_MODEL_NAME,
)


# ── Data Structures ──────────────────────────


@dataclass
class Detection:
    """A single detected person."""

    bbox: np.ndarray                        # [x1, y1, x2, y2]  (pixels)
    confidence: float                       # 0.0 – 1.0
    center: Tuple[float, float]             # (cx, cy)  centre of bbox
    class_name: str = "person"
    class_id: int = 0


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-Maximum Suppression (NMS) for merging candidate boxes."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = (areas[i] + areas[order[1:]] - inter)
        ovr = np.where(denom > 0, inter / denom, 0.0)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ── Detector ─────────────────────────────────


class PersonDetector:
    """
    Wraps a YOLO model with adaptive tiling for person and crowd detection.

    Usage
    -----
    >>> detector = PersonDetector()
    >>> detections = detector.detect(frame)
    >>> annotated  = detector.draw(frame, detections)
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        classes: List[int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0, 25]  # person + overhead umbrella
        self._model: YOLO | None = None

    # ── Model lifecycle ──────────────────────

    def load_model(self) -> None:
        """Download (if needed) and load the YOLO model."""
        self._model = YOLO(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ────────────────────────────

    def detect(self, frame: np.ndarray, adaptive_tiling: bool = True) -> List[Detection]:
        """
        Run person detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV image in BGR format.
        adaptive_tiling : bool
            Enable multi-tile scanning for high-resolution dense crowd scenes.

        Returns
        -------
        List[Detection]
            One `Detection` per person found.
        """
        if not self.is_loaded:
            self.load_model()

        h, w = frame.shape[:2]

        # Use adaptive tiling if high resolution / elevated crowd perspective
        if adaptive_tiling and (h >= 800 or w >= 800):
            grid_r = 3 if h >= 1000 else 2
            grid_c = 2 if w >= 700 else 1

            crop_h = h // grid_r
            crop_w = w // grid_c
            overlap = 40

            all_boxes = []
            all_scores = []

            for r in range(grid_r):
                for c in range(grid_c):
                    y1 = max(0, r * crop_h - overlap)
                    y2 = min(h, (r + 1) * crop_h + overlap)
                    x1 = max(0, c * crop_w - overlap)
                    x2 = min(w, (c + 1) * crop_w + overlap)

                    crop = frame[y1:y2, x1:x2]
                    res = self._model(
                        crop,
                        imgsz=640,
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

            if not all_boxes:
                return []

            boxes_arr = np.array(all_boxes, dtype=np.float32)
            scores_arr = np.array(all_scores, dtype=np.float32)
            keep = nms_boxes(boxes_arr, scores_arr, self.iou_threshold)

            detections: List[Detection] = []
            for i in keep:
                b = boxes_arr[i]
                cx = float((b[0] + b[2]) / 2)
                cy = float((b[1] + b[3]) / 2)
                detections.append(
                    Detection(
                        bbox=b,
                        confidence=float(scores_arr[i]),
                        center=(cx, cy),
                        class_name="person",
                        class_id=PERSON_CLASS_ID,
                    )
                )
            return detections

        # Standard single-shot detection for normal resolution frames
        results = self._model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                detections.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        confidence=conf,
                        center=(cx, cy),
                        class_name="person",
                        class_id=PERSON_CLASS_ID,
                    )
                )

        return detections

    # ── Visualisation ────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        detections: List[Detection],
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.55,
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on a copy of frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(annotated, (int(det.center[0]), int(det.center[1])), 4, (0, 0, 255), -1)

            label = f"{det.confidence:.2f}"
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
