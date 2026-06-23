# src/threat/threat_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────
# Threat Analysis Engine
#
# Analyzes active tracks and predicted trajectories to detect:
#   1. Restricted zone intrusion (current or predicted)
#   2. Loitering (staying in same area too long)
#   3. Speed anomaly (moving too fast or erratically)
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from loguru import logger
from datetime import datetime


# ── Alert dataclass ───────────────────────────────────────────────────────────
@dataclass
class Alert:
    """A single threat alert."""
    track_id:   int
    alert_type: str                         # "ZONE_INTRUSION", "LOITERING", "SPEED_ANOMALY"
    severity:   str                         # "HIGH", "MEDIUM", "LOW"
    message:    str
    position:   Tuple[float, float]
    frame_id:   int
    timestamp:  str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    @property
    def color(self) -> Tuple[int, int, int]:
        return {
            "HIGH":   (0, 0, 255),          # red
            "MEDIUM": (0, 165, 255),        # orange
            "LOW":    (0, 255, 255),        # yellow
        }.get(self.severity, (255, 255, 255))


# ── Threat Analyzer ───────────────────────────────────────────────────────────
class ThreatAnalyzer:
    """
    Analyzes tracks and predictions to generate threat alerts.

    Usage:
        analyzer = ThreatAnalyzer("configs/config.yaml", "configs/zones.json")
        alerts   = analyzer.analyze(tracks, predictions, frame_id)
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        zones_path:  str = "configs/zones.json",
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.threat_cfg = self.config["threat"]

        # Load restricted zones
        self.zones = self._load_zones(zones_path)
        logger.info(f"Threat analyzer ready | {len(self.zones)} restricted zone(s)")

        # Loitering tracker: track_id → list of recent positions
        self.loiter_history: Dict[int, List[Tuple[float, float]]] = {}

        # Alert log
        self.alert_log: List[Alert] = []

        # Active alerts per track (to avoid spamming same alert)
        self.active_alerts: Dict[int, Dict[str, int]] = {}   # {tid: {type: frame_id}}
        self.alert_cooldown = 30                              # frames between same alert

    def _load_zones(self, zones_path: str) -> List[np.ndarray]:
        """Load zones from JSON file as numpy arrays."""
        path = Path(zones_path)
        if not path.exists():
            logger.warning(f"No zones file found at {zones_path}. Run zone_drawer first.")
            return []
        with open(path, "r") as f:
            raw = json.load(f)
        zones = [np.array(z, dtype=np.int32) for z in raw if len(z) >= 3]
        return zones

    def _point_in_zone(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside any restricted zone."""
        pt = (float(point[0]), float(point[1]))
        for zone in self.zones:
            if cv2.pointPolygonTest(zone, pt, False) >= 0:
                return True
        return False

    def _path_enters_zone(self, path: List[Tuple[float, float]]) -> bool:
        """Check if any predicted future point enters a restricted zone."""
        return any(self._point_in_zone(pt) for pt in path)

    def _is_on_cooldown(self, track_id: int, alert_type: str, frame_id: int) -> bool:
        """Prevent same alert from firing every frame."""
        if track_id not in self.active_alerts:
            self.active_alerts[track_id] = {}
        last = self.active_alerts[track_id].get(alert_type, -999)
        return (frame_id - last) < self.alert_cooldown

    def _register_alert(self, alert: Alert):
        """Register alert in cooldown tracker and log."""
        if alert.track_id not in self.active_alerts:
            self.active_alerts[alert.track_id] = {}
        self.active_alerts[alert.track_id][alert.alert_type] = alert.frame_id
        self.alert_log.append(alert)
        logger.warning(f"🚨 ALERT [{alert.severity}] {alert.alert_type} | "
                       f"Track ID:{alert.track_id} | {alert.message}")

    # ── Check 1: Zone intrusion ───────────────────────────────────────────────
    def check_zone_intrusion(
        self,
        track_id: int,
        current_pos: Tuple[float, float],
        predicted_path: Optional[List[Tuple[float, float]]],
        frame_id: int,
    ) -> Optional[Alert]:
        if not self.zones:
            return None

        # Current position already in zone
        if self._point_in_zone(current_pos):
            if not self._is_on_cooldown(track_id, "ZONE_INTRUSION", frame_id):
                alert = Alert(
                    track_id   = track_id,
                    alert_type = "ZONE_INTRUSION",
                    severity   = "HIGH",
                    message    = "Object is inside restricted zone!",
                    position   = current_pos,
                    frame_id   = frame_id,
                )
                self._register_alert(alert)
                return alert

        # Predicted path will enter zone
        if predicted_path and self._path_enters_zone(predicted_path):
            if not self._is_on_cooldown(track_id, "PREDICTED_INTRUSION", frame_id):
                alert = Alert(
                    track_id   = track_id,
                    alert_type = "PREDICTED_INTRUSION",
                    severity   = "MEDIUM",
                    message    = "Object predicted to enter restricted zone!",
                    position   = current_pos,
                    frame_id   = frame_id,
                )
                self._register_alert(alert)
                return alert

        return None

    # ── Check 2: Loitering ────────────────────────────────────────────────────
    def check_loitering(
        self,
        track_id: int,
        positions: List[Tuple[float, float]],
        fps: float,
        frame_id: int,
    ) -> Optional[Alert]:
        threshold_sec    = self.threat_cfg["loitering_threshold_seconds"]
        radius           = self.threat_cfg["loitering_radius_pixels"]
        frames_threshold = int(threshold_sec * fps)

        if len(positions) < frames_threshold:
            return None

        # Check if object has stayed within radius for threshold frames
        recent = positions[-frames_threshold:]
        center = np.mean(recent, axis=0)
        dists  = [np.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2) for p in recent]

        if max(dists) < radius:
            if not self._is_on_cooldown(track_id, "LOITERING", frame_id):
                alert = Alert(
                    track_id   = track_id,
                    alert_type = "LOITERING",
                    severity   = "MEDIUM",
                    message    = f"Object loitering for {threshold_sec}+ seconds!",
                    position   = tuple(center),
                    frame_id   = frame_id,
                )
                self._register_alert(alert)
                return alert

        return None

    # ── Check 3: Speed anomaly ────────────────────────────────────────────────
    def check_speed_anomaly(
        self,
        track_id: int,
        positions: List[Tuple[float, float]],
        frame_id: int,
    ) -> Optional[Alert]:
        if len(positions) < 10:
            return None

        multiplier = self.threat_cfg["speed_anomaly_multiplier"]

        # Compute per-frame speeds
        speeds = [
            np.sqrt((positions[i][0]-positions[i-1][0])**2 +
                    (positions[i][1]-positions[i-1][1])**2)
            for i in range(1, len(positions))
        ]
        avg_speed     = np.mean(speeds[:-1])   # historical average
        current_speed = speeds[-1]             # latest speed

        if avg_speed > 0 and current_speed > avg_speed * multiplier:
            if not self._is_on_cooldown(track_id, "SPEED_ANOMALY", frame_id):
                alert = Alert(
                    track_id   = track_id,
                    alert_type = "SPEED_ANOMALY",
                    severity   = "LOW",
                    message    = f"Sudden speed change! ({current_speed:.1f} px/frame)",
                    position   = positions[-1],
                    frame_id   = frame_id,
                )
                self._register_alert(alert)
                return alert

        return None

    # ── Main analyze method ───────────────────────────────────────────────────
    def analyze(
        self,
        tracks,
        predictions: Dict[int, List[Tuple[float, float]]],
        frame_id: int,
        fps: float = 25.0,
    ) -> List[Alert]:
        """
        Run all threat checks on current tracks and predictions.

        Args:
            tracks:      active Track objects from tracker
            predictions: {track_id: [(x,y)...]} from predictor
            frame_id:    current frame number
            fps:         video fps (for loitering time calculation)

        Returns:
            List of Alert objects triggered this frame
        """
        frame_alerts = []

        for track in tracks:
            tid      = track.track_id
            pos      = track.positions
            curr_pos = pos[-1] if pos else (0, 0)
            pred     = predictions.get(tid)

            # 1. Zone intrusion
            alert = self.check_zone_intrusion(tid, curr_pos, pred, frame_id)
            if alert:
                frame_alerts.append(alert)

            # 2. Loitering
            alert = self.check_loitering(tid, pos, fps, frame_id)
            if alert:
                frame_alerts.append(alert)

            # 3. Speed anomaly
            alert = self.check_speed_anomaly(tid, pos, frame_id)
            if alert:
                frame_alerts.append(alert)

        return frame_alerts

    def save_alert_log(self, path: str = "outputs/alerts/alert_log.json"):
        """Save all alerts to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        log = [
            {
                "timestamp" : a.timestamp,
                "frame_id"  : a.frame_id,
                "track_id"  : a.track_id,
                "alert_type": a.alert_type,
                "severity"  : a.severity,
                "message"   : a.message,
                "position"  : list(a.position),
            }
            for a in self.alert_log
        ]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info(f"Alert log saved → {path} | {len(log)} alerts total")


# ── Visualization helpers ─────────────────────────────────────────────────────
def draw_zones(frame: np.ndarray, zones: List[np.ndarray]) -> np.ndarray:
    """Draw restricted zones on frame."""
    frame = frame.copy()
    for i, zone in enumerate(zones):
        if len(zone) < 3:
            continue
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)
        cx = int(np.mean(zone[:, 0]))
        cy = int(np.mean(zone[:, 1]))
        cv2.putText(frame, f"RESTRICTED ZONE {i+1}", (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_alerts(frame: np.ndarray, alerts: List[Alert]) -> np.ndarray:
    """Draw alert indicators on frame."""
    frame = frame.copy()
    h, w  = frame.shape[:2]

    for alert in alerts:
        # Flash circle at object position
        cx, cy = int(alert.position[0]), int(alert.position[1])
        cv2.circle(frame, (cx, cy), 30, alert.color, 3)
        cv2.circle(frame, (cx, cy), 35, alert.color, 1)

        # Alert label above object
        cv2.putText(frame, f"⚠ {alert.alert_type}",
                    (cx - 60, cy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert.color, 2)

    # Alert panel — top right
    if alerts:
        panel_x = w - 320
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 55), (w, 55 + len(alerts) * 22 + 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"🚨 ACTIVE ALERTS: {len(alerts)}",
                    (panel_x + 5, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for i, alert in enumerate(alerts):
            cv2.putText(frame,
                        f"ID:{alert.track_id} {alert.alert_type} [{alert.severity}]",
                        (panel_x + 5, 90 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, alert.color, 1)

    return frame