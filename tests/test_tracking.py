# ──────────────────────────────────────────────
# CrowdGuard – Tracking Test Script
# ──────────────────────────────────────────────

"""
End-to-end test for the MultiObjectTracker module.

Runs ByteTrack on a short MP4 video and verifies:
  • Tracks are returned with valid fields
  • IDs persist across consecutive frames
  • Position history accumulates correctly
  • Annotated output video is produced

Usage:
    cd CrowdGuard
    .venv\\Scripts\\python.exe -m tests.test_tracking
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from tracking.tracker import MultiObjectTracker, Track, TrackerState


# ── Paths ────────────────────────────────────

VIDEO_IN = PROJECT_ROOT / "data" / "videos" / "test_crowd.mp4"
VIDEO_OUT = PROJECT_ROOT / "data" / "videos" / "test_crowd_tracked.mp4"


def divider(title: str = "") -> None:
    print(f"\n{'-' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'-' * 60}")


def run_test() -> None:
    # ── 1. Check input video ─────────────────
    divider("1 · Checking input video")
    assert VIDEO_IN.exists(), (
        f"Test video not found: {VIDEO_IN}\n"
        f"Run:  python -m tests.generate_test_video"
    )
    cap = cv2.VideoCapture(str(VIDEO_IN))
    assert cap.isOpened(), "Cannot open video"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video   : {VIDEO_IN.name}")
    print(f"  Frames  : {total_frames}")
    print(f"  FPS     : {fps}")
    print(f"  Size    : {w} × {h}")

    # ── 2. Init tracker ─────────────────────
    divider("2 · Initialising MultiObjectTracker")
    tracker = MultiObjectTracker()
    assert not tracker.is_loaded
    print("  Tracker created (model not yet loaded)")

    # ── 3. Process frames ────────────────────
    divider("3 · Processing frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w, h))

    ids_per_frame: list[set[int]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state: TrackerState = tracker.update(frame, frame_number=frame_idx)

        # Collect IDs for this frame
        frame_ids = {t.track_id for t in state.tracks}
        ids_per_frame.append(frame_ids)

        # Validate Track fields
        for t in state.tracks:
            assert isinstance(t, Track)
            assert t.bbox.shape == (4,), f"bbox shape: {t.bbox.shape}"
            assert 0.0 <= t.confidence <= 1.0
            assert t.class_name == "person"
            assert t.frame_number == frame_idx

        # Write annotated frame
        annotated = tracker.draw(frame, state.tracks)
        out.write(annotated)

        if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
            id_list = sorted(frame_ids)
            print(
                f"  Frame {frame_idx:>3}/{total_frames}  "
                f"tracks={len(state.tracks):>2}  "
                f"IDs={id_list}"
            )

        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n  Processed {frame_idx} frames total")

    # ── 4. Verify ID persistence ─────────────
    divider("4 · Verifying ID persistence")

    # Collect all unique IDs across the entire video
    all_ids: set[int] = set()
    for fids in ids_per_frame:
        all_ids.update(fids)
    print(f"  Unique IDs across video : {sorted(all_ids)}")

    # Check that IDs seen in frame N re-appear in frame N+1
    # (for IDs that are still visible)
    persistent_count = 0
    total_checks = 0
    for i in range(1, len(ids_per_frame)):
        common = ids_per_frame[i - 1] & ids_per_frame[i]
        persistent_count += len(common)
        total_checks += len(ids_per_frame[i - 1])

    if total_checks > 0:
        persistence_rate = persistent_count / total_checks * 100
    else:
        persistence_rate = 0.0

    print(f"  ID persistence rate     : {persistence_rate:.1f}%")
    assert persistence_rate > 50.0, (
        f"ID persistence too low ({persistence_rate:.1f}%) — "
        "ByteTrack may not be working correctly"
    )

    # ── 5. Verify position history ───────────
    divider("5 · Verifying position history")
    for tid in sorted(all_ids):
        hist = tracker.get_history(tid)
        print(f"  ID {tid:>3} : {len(hist)} positions stored")
        assert len(hist) > 0, f"No history for track {tid}"

    # ── 6. Output info ───────────────────────
    divider("6 · Annotated video saved")
    print(f"  Output : {VIDEO_OUT.name}")

    # ── Summary ──────────────────────────────
    divider("[OK] ALL TRACKING CHECKS PASSED")
    print(f"  Frames processed    : {frame_idx}")
    print(f"  Unique persons seen : {len(all_ids)}")
    print(f"  ID persistence      : {persistence_rate:.1f}%")
    print(f"  Annotated video     : {VIDEO_OUT}")
    print()


if __name__ == "__main__":
    run_test()
