"""Concurrent live-camera capture with shared-model, per-camera crowd analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from analytics.anomaly import CounterFlowDetector
from analytics.bottleneck import BottleneckDetector, assign_zone
from analytics.cpwp import CrowdPressureWavePredictor
from analytics.density import DensityAnalyser, classify_density
from analytics.flow import FlowAnalyser
from config import DENSITY_ZONES, YOLO_MODEL_NAME
from detection.gate_detector import GateDetector
from risk.risk_engine import RiskEngine
from routing.route_engine import RouteEngine
from tracking.tracker import MultiObjectTracker

_RISK_ORDER = {"SAFE": 0, "NORMAL": 1, "WARNING": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass
class CameraSession:
    id: str
    name: str
    source: int | str
    area: str
    capture: cv2.VideoCapture
    tracker: MultiObjectTracker
    gate_detector: GateDetector = field(default_factory=GateDetector)
    density_analyser: DensityAnalyser = field(default_factory=DensityAnalyser)
    flow: FlowAnalyser = field(default_factory=FlowAnalyser)
    counter_flow: CounterFlowDetector = field(default_factory=CounterFlowDetector)
    bottleneck: BottleneckDetector = field(default_factory=BottleneckDetector)
    pressure_predictor: CrowdPressureWavePredictor = field(default_factory=CrowdPressureWavePredictor)
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    latest_frame: bytes | None = None
    status: str = "CONNECTING"
    error: str | None = None
    frame_index: int = 0
    people: int = 0
    density: float = 0.0
    risk_level: str = "SAFE"
    risk_score: float = 0.0
    dominant_flow: str = "STATIONARY"
    gate_count: int = 0
    route: dict[str, Any] = field(default_factory=dict)
    last_processed: float = 0.0


class CameraManager:
    """Manage live camera feeds and independently analyze each camera area."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._cameras: dict[str, CameraSession] = {}
        self._next_id = 1
        self._model: YOLO | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._route_engine = RouteEngine()
        self._route_engine.load_graph()

    @staticmethod
    def _open_capture(source: int | str) -> cv2.VideoCapture:
        if isinstance(source, int):
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                ],
            )
        if not capture.isOpened():
            capture.release()
            if isinstance(source, int):
                raise ValueError(
                    "Camera device could not be opened; verify its index and backend access."
                )
            raise ValueError(
                "Network stream could not be opened. Verify its playback URL, credentials, "
                "and camera-server playback/IP permissions."
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def add_camera(self, name: str, source: int | str, area: str) -> dict[str, Any]:
        clean_name = name.strip()[:48]
        if not clean_name:
            raise ValueError("Camera name is required.")
        if area not in DENSITY_ZONES:
            raise ValueError("Choose a valid venue area for this camera.")

        capture = self._open_capture(source)
        try:
            with self._lock:
                if self._model is None:
                    self._model = YOLO(YOLO_MODEL_NAME)
                camera_id = f"camera-{self._next_id}"
                self._next_id += 1
                camera = CameraSession(
                    id=camera_id,
                    name=clean_name,
                    source=source,
                    area=area,
                    capture=capture,
                    tracker=MultiObjectTracker(model=self._model),
                )
                self._cameras[camera_id] = camera
                if self._thread is None or not self._thread.is_alive():
                    self._stop.clear()
                    self._thread = threading.Thread(
                        target=self._process_cameras,
                        name="crowdguard-live-cameras",
                        daemon=True,
                    )
                    self._thread.start()
                return self._camera_payload(camera)
        except Exception:
            capture.release()
            raise

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            camera = self._cameras.pop(camera_id, None)
        if camera is None:
            return False
        with camera.frame_lock:
            camera.status = "STOPPED"
            camera.capture.release()
            camera.latest_frame = None
        return True

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            cameras = list(self._cameras.values())
            self._cameras.clear()
        for camera in cameras:
            with camera.frame_lock:
                camera.status = "STOPPED"
                camera.capture.release()
                camera.latest_frame = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        with self._lock:
            return any(camera.status in {"ANALYZING", "CONNECTING"} for camera in self._cameras.values())

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._camera_payload(camera) for camera in self._cameras.values()]

    def get_frame(self, camera_id: str) -> bytes | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
        if camera is None:
            return None
        with camera.frame_lock:
            return camera.latest_frame

    def dashboard_state(self) -> dict[str, Any]:
        with self._lock:
            cameras = [self._camera_payload(camera) for camera in self._cameras.values()]
        directions = [
            {
                "camera_id": camera["id"],
                "camera": camera["name"],
                "area": camera["area"],
                "people": camera["people"],
                "density": camera["density"],
                "risk_level": camera["risk_level"],
                "direction": camera["route"].get("direction", "No safe route available"),
                "path": camera["route"].get("path", []),
                "exit": camera["route"].get("exit"),
                "explanation": camera["route"].get("explanation", ""),
            }
            for camera in cameras
            if camera["status"] == "ANALYZING"
        ]
        return {
            "cameras": cameras,
            "camera_count": len(cameras),
            "analyzing_count": sum(camera["status"] == "ANALYZING" for camera in cameras),
            "total_people": sum(camera["people"] for camera in cameras),
            "directions": directions,
            "recommendation_note": (
                "Directions use each camera's assigned venue area and the configured venue map."
                if directions else
                "Add cameras and assign each camera to its venue area to calculate safe routes."
            ),
        }

    @staticmethod
    def _camera_payload(camera: CameraSession) -> dict[str, Any]:
        with camera.frame_lock:
            frame_ready = camera.latest_frame is not None
        return {
            "id": camera.id,
            "name": camera.name,
            "area": camera.area,
            "status": camera.status,
            "error": camera.error,
            "people": camera.people,
            "density": round(camera.density, 1),
            "risk_level": camera.risk_level,
            "risk_score": round(camera.risk_score, 1),
            "dominant_flow": camera.dominant_flow,
            "gate_count": camera.gate_count,
            "frame_index": camera.frame_index,
            "frame_ready": frame_ready,
            "route": camera.route,
        }

    def _process_cameras(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                cameras = list(self._cameras.values())
            if not cameras:
                self._stop.wait(0.1)
                continue
            for camera in cameras:
                if self._stop.is_set():
                    break
                try:
                    self._process_camera_frame(camera)
                except Exception as error:
                    camera.status = "ERROR"
                    camera.error = str(error)[:240]
                    print(f"[CAMERA ERROR] {camera.name}: {error}")
            self._stop.wait(0.005)

    def _process_camera_frame(self, camera: CameraSession) -> None:
        with camera.frame_lock:
            if camera.status == "STOPPED":
                return
            ok, frame = camera.capture.read()
            if not ok:
                camera.status = "DISCONNECTED"
                camera.error = "No frames received from this camera."
                return

        camera.status = "ANALYZING"
        camera.error = None
        height, width = frame.shape[:2]

        with self._inference_lock:
            tracker_state = camera.tracker.update(frame, camera.frame_index)
        camera.gate_detector.update(frame, tracker_state.tracks)
        centers = [(track.center_x, track.center_y) for track in tracker_state.tracks]
        zones = camera.density_analyser.compute(centers, (height, width))
        flows = camera.flow.compute_person_flows(tracker_state.history)
        zone_flows = camera.flow.compute_zone_flows(flows, (height, width))
        counter_flow_reports = camera.counter_flow.analyse(zone_flows)
        track_zones = {
            track.track_id: assign_zone(track.center_x, track.center_y, (height, width))
            for track in tracker_state.tracks
        }
        camera.bottleneck.update(track_zones, camera.frame_index)
        gate_reports = camera.bottleneck.get_reports()
        risk = camera.risk_engine.evaluate(
            zone_reports=zones,
            counter_flow_reports=counter_flow_reports,
            gate_reports=gate_reports,
            zone_flows=zone_flows,
        )
        camera.pressure_predictor.predict(zones, zone_flows)

        camera.people = len(tracker_state.tracks)
        zone_capacity = DENSITY_ZONES[camera.area]["capacity"]
        camera.density = camera.people / max(zone_capacity, 1) * 100
        camera.risk_level = risk.risk_level
        camera.risk_score = float(risk.risk_score)
        directions = [
            flow.direction for flow in flows if flow.direction != "STATIONARY"
        ]
        camera.dominant_flow = max(set(directions), key=directions.count) if directions else "STATIONARY"
        camera.gate_count = len(camera.gate_detector.to_api_dict(width, height))
        self._update_route_recommendations(cameras_snapshot=None)

        annotated = camera.gate_detector.draw(frame)
        for track in tracker_state.tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 212, 0), 2)
            cv2.putText(
                annotated, f"ID-{track.track_id}", (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 212, 0), 1, cv2.LINE_AA,
            )
        cv2.putText(
            annotated, f"{camera.name} | {camera.people} people | {camera.risk_level}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 255), 2, cv2.LINE_AA,
        )
        encoded, jpeg = cv2.imencode(".jpg", annotated)
        if encoded:
            with camera.frame_lock:
                if camera.status == "STOPPED":
                    return
                camera.latest_frame = jpeg.tobytes()
        camera.frame_index += 1
        camera.last_processed = time.time()

    def _update_route_recommendations(self, cameras_snapshot: list[CameraSession] | None) -> None:
        if cameras_snapshot is None:
            with self._lock:
                cameras_snapshot = list(self._cameras.values())
        area_levels: dict[str, str] = {}
        for area in DENSITY_ZONES:
            area_levels[area] = "SAFE"
        for camera in cameras_snapshot:
            level = classify_density(camera.density / 100)
            previous = area_levels.get(camera.area, "SAFE")
            if _RISK_ORDER.get(level, 0) >= _RISK_ORDER.get(previous, 0):
                area_levels[camera.area] = level
        self._route_engine.update_risk_state({
            area.upper().replace(" ", "_"): level
            for area, level in area_levels.items()
        })

        for camera in cameras_snapshot:
            source = camera.area.upper().replace(" ", "_")
            try:
                route = self._route_engine.find_safest_route(source)
                path = route.route_nodes
                camera.route = {
                    "source": camera.area,
                    "exit": route.recommended_exit,
                    "path": path,
                    "direction": (
                        f"Move from {camera.area} toward {route.recommended_exit}"
                        if path else "No safe route available"
                    ),
                    "explanation": route.explanation.splitlines()[-1] if path else route.explanation,
                }
            except (ValueError, RuntimeError) as error:
                camera.route = {
                    "source": camera.area,
                    "exit": None,
                    "path": [],
                    "direction": "No safe route available",
                    "explanation": str(error),
                }
