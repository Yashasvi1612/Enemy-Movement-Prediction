# src/threat/zone_drawer.py
# ─────────────────────────────────────────────────────────────────────────────
# Interactive Restricted Zone Drawer
#
# Shows the first frame of the video and lets you draw restricted zones
# by clicking points with the mouse. Press Enter to finish a zone,
# press 'n' for a new zone, press 's' to save and exit.
#
# Usage:
#   python -m src.threat.zone_drawer --source data/raw/test.mp4
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from loguru import logger


class ZoneDrawer:
    """
    Interactive tool to draw restricted zones on a video frame.

    Controls:
        Left click  — add a point to current zone
        Enter       — close and finish current zone
        N           — start a new zone
        Z           — undo last point
        C           — clear current zone
        S           — save all zones and exit
        Q           — quit without saving
    """

    def __init__(self, frame: np.ndarray, existing_zones: list = None):
        self.original_frame = frame.copy()
        self.frame          = frame.copy()
        self.zones          = existing_zones or []     # completed zones
        self.current_zone   = []                       # points of zone being drawn
        self.hover_pt       = None
        self.window_name    = "Draw Restricted Zones"

        # Colors
        self.zone_color     = (0, 0, 255)              # red for restricted zones
        self.point_color    = (0, 255, 255)            # yellow for points
        self.preview_color  = (0, 128, 255)            # orange for preview line
        self.fill_alpha     = 0.25

    def _redraw(self):
        """Redraw everything on a fresh copy of the frame."""
        self.frame = self.original_frame.copy()

        # Draw completed zones
        for i, zone in enumerate(self.zones):
            if len(zone) >= 3:
                pts = np.array(zone, dtype=np.int32)

                # Filled semi-transparent overlay
                overlay = self.frame.copy()
                cv2.fillPoly(overlay, [pts], self.zone_color)
                cv2.addWeighted(overlay, self.fill_alpha, self.frame, 1 - self.fill_alpha, 0, self.frame)

                # Border
                cv2.polylines(self.frame, [pts], isClosed=True, color=self.zone_color, thickness=2)

                # Zone label
                cx = int(np.mean([p[0] for p in zone]))
                cy = int(np.mean([p[1] for p in zone]))
                cv2.putText(self.frame, f"ZONE {i+1}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw current zone being drawn
        for pt in self.current_zone:
            cv2.circle(self.frame, pt, 5, self.point_color, -1)

        if len(self.current_zone) >= 2:
            pts = np.array(self.current_zone, dtype=np.int32)
            cv2.polylines(self.frame, [pts], isClosed=False, color=self.point_color, thickness=2)

        # Preview line to mouse cursor
        if self.current_zone and self.hover_pt:
            cv2.line(self.frame, self.current_zone[-1], self.hover_pt, self.preview_color, 1)
            if len(self.current_zone) >= 2:
                cv2.line(self.frame, self.current_zone[0], self.hover_pt, self.preview_color, 1)

        # Instructions overlay
        self._draw_instructions()

        cv2.imshow(self.window_name, self.frame)

    def _draw_instructions(self):
        """Draw instruction panel on frame."""
        h, w = self.frame.shape[:2]
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (0, h - 130), (320, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, self.frame, 0.3, 0, self.frame)

        instructions = [
            "LEFT CLICK  — add point",
            "ENTER       — finish zone",
            "N           — new zone",
            "Z           — undo point",
            "C           — clear zone",
            "S           — save & exit",
            "Q           — quit",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.frame, text, (8, h - 115 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)

        # Status bar
        status = f"Zones: {len(self.zones)} complete | Current: {len(self.current_zone)} points"
        cv2.putText(self.frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_pt = (x, y)
            self._redraw()

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone.append((x, y))
            logger.info(f"Point added: ({x}, {y}) | Zone has {len(self.current_zone)} points")
            self._redraw()

    def _finish_zone(self):
        """Close current zone and add to completed zones."""
        if len(self.current_zone) >= 3:
            self.zones.append(self.current_zone.copy())
            logger.info(f"Zone {len(self.zones)} completed with {len(self.current_zone)} points")
            self.current_zone = []
        else:
            logger.warning("Need at least 3 points to create a zone!")
        self._redraw()

    def draw(self) -> list:
        """
        Open interactive window for zone drawing.
        Returns list of completed zones (each zone is list of (x,y) points).
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()

        logger.info("Zone drawer open. Click to add points, Enter to finish zone, S to save.")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == 13:           # Enter — finish current zone
                self._finish_zone()

            elif key == ord('n'):   # N — new zone (finish current first)
                self._finish_zone()

            elif key == ord('z'):   # Z — undo last point
                if self.current_zone:
                    self.current_zone.pop()
                    self._redraw()

            elif key == ord('c'):   # C — clear current zone
                self.current_zone = []
                self._redraw()

            elif key == ord('s'):   # S — save and exit
                self._finish_zone()
                if self.zones:
                    logger.info(f"Saving {len(self.zones)} zone(s)")
                cv2.destroyWindow(self.window_name)
                return self.zones

            elif key == ord('q'):   # Q — quit without saving
                logger.info("Quit without saving.")
                cv2.destroyWindow(self.window_name)
                return []

        return self.zones


def draw_zones_on_video(source: str, save_path: str = "configs/zones.json") -> list:
    """
    Open first frame of video for zone drawing and save result.

    Args:
        source:    path to video file
        save_path: where to save zone coordinates as JSON

    Returns:
        List of zones
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {source}")
        return []

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Cannot read first frame")
        return []

    logger.info(f"Video frame: {frame.shape[1]}x{frame.shape[0]}")
    logger.info("Opening zone drawer...")

    # Load existing zones if any
    existing = []
    if Path(save_path).exists():
        with open(save_path, "r") as f:
            existing = json.load(f)
        logger.info(f"Loaded {len(existing)} existing zones")

    drawer = ZoneDrawer(frame, existing)
    zones  = drawer.draw()

    if zones:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(zones, f, indent=2)
        logger.info(f"Zones saved → {save_path}")
    else:
        logger.warning("No zones saved.")

    return zones


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw restricted zones on video")
    parser.add_argument("--source", type=str, required=True, help="Video file path")
    parser.add_argument("--output", type=str, default="configs/zones.json", help="Save zones to")
    args = parser.parse_args()

    zones = draw_zones_on_video(args.source, args.output)
    print(f"\n✅ {len(zones)} zone(s) saved to {args.output}")
    for i, z in enumerate(zones):
        print(f"  Zone {i+1}: {len(z)} points → {z}")