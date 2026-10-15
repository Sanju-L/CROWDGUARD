"""Deterministic tests for geometric gate localization and temporal IDs."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from detection.gate_detector import GateDetector


def doorway_frame(*, gate_x: int = 220, occluded: bool = False) -> np.ndarray:
    frame = np.full((480, 640, 3), 115, dtype=np.uint8)
    color = (245, 245, 245)
    cv2.line(frame, (gate_x, 100), (gate_x, 410), color, 5)
    cv2.line(frame, (gate_x + 150, 100), (gate_x + 150, 410), color, 5)
    cv2.line(frame, (gate_x, 100), (gate_x + 150, 100), color, 5)
    if occluded:
        cv2.rectangle(frame, (gate_x + 125, 175), (gate_x + 170, 360), (20, 20, 20), -1)
    return frame


class GateDetectorTests(unittest.TestCase):
    def test_plain_scene_does_not_create_gate(self) -> None:
        detector = GateDetector()
        frame = np.full((480, 640, 3), 115, dtype=np.uint8)
        for _ in range(30):
            detector.update(frame)
        self.assertEqual(detector.to_api_dict(640, 480), [])

    def test_shelving_with_multiple_crossbars_is_not_reported_as_a_gate(self) -> None:
        detector = GateDetector()
        shelf = np.full((480, 640, 3), 115, dtype=np.uint8)
        for x in (220, 370):
            cv2.line(shelf, (x, 100), (x, 410), (245, 245, 245), 5)
        for y in (100, 200, 300, 400):
            cv2.line(shelf, (220, y), (370, y), (245, 245, 245), 5)

        for _ in range(21):
            detector.update(shelf)
        self.assertEqual(detector.to_api_dict(640, 480), [])

    def test_table_sized_frame_is_not_reported_as_a_gate(self) -> None:
        detector = GateDetector()
        table = np.full((480, 640, 3), 115, dtype=np.uint8)
        cv2.line(table, (220, 190), (220, 330), (245, 245, 245), 5)
        cv2.line(table, (370, 190), (370, 330), (245, 245, 245), 5)
        cv2.line(table, (220, 190), (370, 190), (245, 245, 245), 5)

        for _ in range(21):
            detector.update(table)
        self.assertEqual(detector.to_api_dict(640, 480), [])

    def test_doorway_geometry_is_found_and_keeps_image_boundaries(self) -> None:
        detector = GateDetector()
        frame = doorway_frame()
        for _ in range(16):
            detector.update(frame)

        gates = detector.to_api_dict(640, 480)
        self.assertEqual(len(gates), 1)
        gate = gates[0]
        self.assertEqual(gate["id"], "Gate 1")
        self.assertEqual(gate["coordinate_space"], "camera-image-normalized")
        self.assertEqual(
            gate["confidence_basis"],
            "structural_geometry_and_temporal_persistence",
        )
        self.assertEqual(len(gate["corners"]), 4)
        self.assertTrue(all(
            0 <= coordinate <= 1
            for corner in gate["corners"]
            for coordinate in corner
        ))
        self.assertTrue(0.25 < gate["cx"] < 0.6)
        self.assertGreater(gate["frames_seen"], 1)

    def test_gate_id_is_stable_and_distinct_openings_get_distinct_ids(self) -> None:
        detector = GateDetector()
        first_frame = doorway_frame()
        two_gates = first_frame.copy()
        cv2.line(two_gates, (440, 100), (440, 410), (245, 245, 245), 5)
        cv2.line(two_gates, (590, 100), (590, 410), (245, 245, 245), 5)
        cv2.line(two_gates, (440, 100), (590, 100), (245, 245, 245), 5)

        for _ in range(31):
            detector.update(two_gates)

        gates = detector.to_api_dict(640, 480)
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate["id"] for gate in gates], ["Gate 1", "Gate 2"])

    def test_temporarily_occluded_gate_is_retained_with_lower_confidence(self) -> None:
        detector = GateDetector()
        visible = doorway_frame()
        for _ in range(16):
            detector.update(visible)
        before = detector.to_api_dict(640, 480)[0]["confidence"]

        for _ in range(10):
            detector.update(doorway_frame(occluded=True))

        gates = detector.to_api_dict(640, 480)
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0]["id"], "Gate 1")
        self.assertTrue(gates[0]["occluded"])
        self.assertLess(gates[0]["confidence"], before)


if __name__ == "__main__":
    unittest.main()
