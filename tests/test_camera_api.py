"""HTTP validation and lifecycle tests for live-camera API endpoints."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app as crowdguard_app


class CameraApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = crowdguard_app.app.test_client()

    @patch.object(crowdguard_app.camera_manager, "dashboard_state", return_value={
        "cameras": [],
        "camera_count": 0,
        "analyzing_count": 0,
        "total_people": 0,
        "directions": [],
        "recommendation_note": "Add a camera",
    })
    def test_list_cameras_returns_dashboard_state(self, _dashboard_state) -> None:
        response = self.client.get("/api/cameras")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["camera_count"], 0)

    @patch.object(crowdguard_app.camera_manager, "add_camera", return_value={
        "id": "camera-1",
        "name": "Entrance",
    })
    def test_add_network_stream_with_area(self, add_camera) -> None:
        response = self.client.post("/api/cameras", json={
            "name": "Entrance",
            "source_type": "stream",
            "source": "rtsp://camera.local/live",
            "area": "Zone A",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["camera"]["id"], "camera-1")
        add_camera.assert_called_once_with(
            name="Entrance",
            source="rtsp://camera.local/live",
            area="Zone A",
        )

    @patch.object(crowdguard_app.camera_manager, "add_camera")
    def test_invalid_camera_inputs_are_rejected(self, add_camera) -> None:
        invalid_sources = [
            {"source_type": "stream", "source": "ftp://camera.local/live"},
            {"source_type": "stream", "source": "rtsp://"},
            {"source_type": "device", "source": "-1"},
        ]
        for source in invalid_sources:
            with self.subTest(source=source):
                response = self.client.post("/api/cameras", json={
                    "name": "Entrance",
                    "area": "Zone A",
                    **source,
                })
                self.assertEqual(response.status_code, 400)
        add_camera.assert_not_called()


if __name__ == "__main__":
    unittest.main()
