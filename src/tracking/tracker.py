# src/tracking/tracker.py
# Phase 2 — Object Tracking using DeepSORT

import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from loguru import logger

from src.detection.detector import Detection


@dataclass
class Track:
    """A single tracked object with full history."""
    track_id:   int
    class_name: str
    bbox:       Tuple[int, int, int, int]
    confidence: float
    frame_id:   int
    positions:  List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[int]                 = field(default_factory=list)
    is_confirmed: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        return self.positions

    @property
    def trajectory_length(self) -> int:
        return len(self.positions)

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    def current_speed(self) -> float:
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


class ObjectTracker:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config       = self._load_config(config_path)
        self.trk_cfg      = self.config["tracking"]
        self.tracker      = self._init_tracker()
        self.track_history: Dict[int, Track] = {}
        self._id_to_class: Dict[int, str]    = {}
        logger.info("Tracker initialized | DeepSORT")

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            return DeepSort(
                max_age=self.trk_cfg["max_age"],
                n_init=self.trk_cfg["min_hits"],
                max_cosine_distance=self.trk_cfg["max_cosine_distance"],
                nn_budget=None,
                embedder="mobilenet",
                half=False,
                bgr=True,
            )
        except ImportError:
            logger.error("deep_sort_realtime not installed. Run: pip install deep-sort-realtime")
            raise

    def _detections_to_deepsort_fmt(self, detections: List[Detection]) -> List[Tuple]:
        results = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            results.append(([x1, y1, w, h], det.confidence, det.class_name))
        return results

    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int) -> List[Track]:
        if not detections:
            raw_tracks = self.tracker.update_tracks([], frame=frame)
        else:
            ds_input   = self._detections_to_deepsort_fmt(detections)
            raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        active_tracks = []
        for raw in raw_tracks:
            if not raw.is_confirmed():
                continue

            tid        = int(raw.track_id)
            ltrb       = raw.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cx, cy     = (x1 + x2) / 2, (y1 + y2) / 2
            cls_name   = raw.get_det_class() or "unknown"

            if tid not in self.track_history:
                self.track_history[tid] = Track(
                    track_id=tid, class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=raw.get_det_conf() or 0.0,
                    frame_id=frame_id, is_confirmed=True,
                )
            else:
                t = self.track_history[tid]
                t.bbox       = (x1, y1, x2, y2)
                t.frame_id   = frame_id
                t.confidence = raw.get_det_conf() or t.confidence

            track = self.track_history[tid]
            track.positions.append((cx, cy))
            track.timestamps.append(frame_id)
            active_tracks.append(track)

        return active_tracks

    def get_trajectory(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        if track_id in self.track_history:
            return self.track_history[track_id].positions
        return None

    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        return {tid: t.positions for tid, t in self.track_history.items()}

    def draw(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        import cv2
        frame  = frame.copy()
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]
        for track in tracks:
            color        = colors[track.track_id % len(colors)]
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            trail = track.positions[-30:]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                pt1   = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2   = (int(trail[i][0]),   int(trail[i][1]))
                cv2.line(frame, pt1, pt2, color, max(1, int(alpha * 3)))
        return frame