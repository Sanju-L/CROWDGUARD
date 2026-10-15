"""Live-camera manager tests with deterministic in-memory camera feeds."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

import numpy as np

import tracking.camera_manager as camera_manager_module
from tracking.camera_manager import CameraManager


class FakeCapture:
    def __init__(self, *_args, **_kwargs) -> None:
        self.frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def set(self, *_args) -> None:
        return None

    def read(self):
        if not self.opened:
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.opened = False


class EmptyTracker:
    def __init__(self, **_kwargs) -> None:
        pass

    def update(self, _frame, _frame_index):
        return type("TrackerState", (), {"tracks": [], "history": {}})()


class CameraManagerTests(unittest.TestCase):
    def test_multiple_camera_feeds_are_analyzed_and_have_area_routes(self) -> None:
        with (
            patch.object(camera_manager_module, "YOLO", return_value=object()),
            patch.object(camera_manager_module.cv2, "VideoCapture", FakeCapture),
            patch.object(camera_manager_module, "MultiObjectTracker", EmptyTracker),
        ):
            manager = CameraManager()
            try:
                manager.add_camera("North", "rtsp://camera/north", "Zone A")
                manager.add_camera("South", "http://camera/south/stream", "Zone C")

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = manager.dashboard_state()
                    if state["analyzing_count"] == 2:
                        break
                    time.sleep(0.05)

                self.assertEqual(state["camera_count"], 2)
                self.assertEqual(state["analyzing_count"], 2)
                self.assertEqual(len(state["directions"]), 2)
                self.assertTrue(all(item["path"] for item in state["directions"]))
                self.assertTrue(all(camera["frame_ready"] for camera in state["cameras"]))
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
