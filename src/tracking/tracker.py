# src/tracking/tracker.py
# Phase 2 — Object Tracking using DeepSORT
# ─────────────────────────────────────────────────────────────────────────────
# Assigns consistent IDs to detected objects across frames.
# Builds and maintains trajectory history per tracked object.
# Output: list of Track objects with ID, position history, class.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from loguru import logger

from src.detection.detector import Detection


# ── Track dataclass ───────────────────────────────────────────────────────────
@dataclass
class Track:
    """A single tracked object with full history."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]       # current bounding box
    confidence: float
    frame_id: int

    # History built up over time
    positions: List[Tuple[float, float]] = field(default_factory=list)   # (cx, cy)
    timestamps: List[int] = field(default_factory=list)                  # frame_ids
    is_confirmed: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Full position history as list of (x, y) points."""
        return self.positions

    @property
    def trajectory_length(self) -> int:
        return len(self.positions)

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    def current_speed(self) -> float:
        """Pixels per frame over last 5 frames."""
        if len(self.positions) < 2:
            return 0.0
        recent = self.positions[-min(5, len(self.positions)):]
        if len(recent) < 2:
            return 0.0
        dists = [
            np.sqrt((recent[i][0] - recent[i-1][0])**2 +
                    (recent[i][1] - recent[i-1][1])**2)
            for i in range(1, len(recent))
        ]
        return float(np.mean(dists))


# ── Tracker class ─────────────────────────────────────────────────────────────
class ObjectTracker:
    """
    DeepSORT-based multi-object tracker.

    Wraps deep_sort_realtime to produce Track objects with full
    position history — ready for the trajectory prediction module.

    Usage:
        tracker = ObjectTracker(config_path="configs/config.yaml")
        tracks = tracker.update(detections, frame, frame_id)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.trk_cfg = self.config["tracking"]
        self.tracker = self._init_tracker()

        # Track history: track_id → Track
        self.track_history: Dict[int, Track] = {}

        # Class mapping from last detection for each track id
        self._id_to_class: Dict[int, str] = {}

        logger.info("Tracker initialized | DeepSORT")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _init_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            return DeepSort(
                max_age=self.trk_cfg["max_age"],
                n_init=self.trk_cfg["min_hits"],
                max_cosine_distance=self.trk_cfg["max_cosine_distance"],
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=False,
                bgr=True,
            )
        except ImportError:
            logger.error("deep_sort_realtime not installed. Run: pip install deep-sort-realtime")
            raise

    def _detections_to_deepsort_fmt(
        self, detections: List[Detection]
    ) -> List[Tuple]:
        """
        Convert Detection objects to DeepSORT input format.
        DeepSORT expects: ([x, y, w, h], confidence, class_name)
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            results.append(([x1, y1, w, h], det.confidence, det.class_name))
        return results

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_id: int
    ) -> List[Track]:
        """
        Update tracker with new detections for this frame.

        Args:
            detections: output from ObjectDetector.detect()
            frame: current BGR frame (needed by DeepSORT for appearance embedding)
            frame_id: current frame index

        Returns:
            List of active Track objects (confirmed tracks only)
        """
        if not detections:
            # Still update tracker with empty list so it ages existing tracks
            raw_tracks = self.tracker.update_tracks([], frame=frame)
        else:
            ds_input = self._detections_to_deepsort_fmt(detections)

            # Map class names by closest bbox match for later lookup
            for det in detections:
                cx = (det.bbox[0] + det.bbox[2]) / 2
                cy = (det.bbox[1] + det.bbox[3]) / 2
                key = (round(cx), round(cy))
                self._id_to_class[key] = det.class_name

            raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        active_tracks = []
        for raw in raw_tracks:
            if not raw.is_confirmed():
                continue

            tid = int(raw.track_id)
            ltrb = raw.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Resolve class name
            cls_name = raw.get_det_class() or "unknown"

            # Update or create Track
            if tid not in self.track_history:
                self.track_history[tid] = Track(
                    track_id=tid,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=raw.get_det_conf() or 0.0,
                    frame_id=frame_id,
                    is_confirmed=True,
                )
            else:
                t = self.track_history[tid]
                t.bbox = (x1, y1, x2, y2)
                t.frame_id = frame_id
                t.confidence = raw.get_det_conf() or t.confidence

            # Append position to history
            track = self.track_history[tid]
            track.positions.append((cx, cy))
            track.timestamps.append(frame_id)
            active_tracks.append(track)

        return active_tracks

    def get_trajectory(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """Return full position history for a track ID."""
        if track_id in self.track_history:
            return self.track_history[track_id].positions
        return None

    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return position histories for all ever-seen tracks."""
        return {tid: t.positions for tid, t in self.track_history.items()}

    def draw(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw tracked objects and their trail on frame."""
        import cv2
        frame = frame.copy()
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]

        for track in tracks:
            color = colors[track.track_id % len(colors)]
            x1, y1, x2, y2 = track.bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID label
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Trail (last 30 positions)
            trail = track.positions[-30:]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]),   int(trail[i][1]))
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pt1, pt2, color, thickness)

        return frame