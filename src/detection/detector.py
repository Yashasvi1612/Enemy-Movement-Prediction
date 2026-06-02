# src/detection/detector.py
# Phase 1 — Object Detection using YOLOv8
# ─────────────────────────────────────────────────────────────────────────────
# Detects people and vehicles in each video frame.
# Output: list of Detection objects with bounding box, class, confidence.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from loguru import logger


# ── Detection dataclass ───────────────────────────────────────────────────────
@dataclass
class Detection:
    """Single object detection result from one frame."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    frame_id: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


# ── Detector class ────────────────────────────────────────────────────────────
class ObjectDetector:
    """
    YOLOv8-based object detector for surveillance footage.

    Usage:
        detector = ObjectDetector(config_path="configs/config.yaml")
        detections = detector.detect(frame, frame_id=0)
    """

    # COCO class names for the classes we care about
    CLASS_NAMES = {
        0: "person",
        2: "car",
        3: "motorbike",
        5: "bus",
        7: "truck"
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.det_cfg = self.config["detection"]
        self.model = self._load_model()
        logger.info(f"Detector initialized | model: {self.det_cfg['model_size']} | device: {self.det_cfg['device']}")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.det_cfg["model_size"])
            return model
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image array (from cv2.imread or VideoCapture)
            frame_id: frame index for logging

        Returns:
            List of Detection objects
        """
        results = self.model(
            frame,
            conf=self.det_cfg["confidence_threshold"],
            iou=self.det_cfg["iou_threshold"],
            classes=self.det_cfg["classes"],
            device=self.det_cfg["device"],
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_id=frame_id
                ))

        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame for visualization."""
        colors = {
            "person":    (0, 255, 100),
            "car":       (255, 165, 0),
            "motorbike": (0, 165, 255),
            "bus":       (255, 0, 165),
            "truck":     (165, 0, 255),
        }
        frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return frame


# ── Video runner (standalone use) ─────────────────────────────────────────────
def run_on_video(source: str, config_path: str, save_output: bool = False):
    """
    Run detection on a video file or webcam.

    Args:
        source: path to video file, or "0" for webcam
        config_path: path to config.yaml
        save_output: whether to save annotated video
    """
    detector = ObjectDetector(config_path)

    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {w}x{h} @ {fps:.1f} fps")

    writer = None
    if save_output:
        out_path = "outputs/visualizations/detection_output.mp4"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        logger.info(f"Saving output to: {out_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, frame_id)
        annotated  = detector.draw(frame, detections)

        # HUD overlay
        cv2.putText(annotated, f"Frame: {frame_id} | Detections: {len(detections)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)

        cv2.imshow("Defense Surveillance — Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Stopped by user.")
            break

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info(f"Done. Processed {frame_id} frames.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on video")
    parser.add_argument("--source", type=str, default="0", help="Video path or '0' for webcam")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--save",   action="store_true", help="Save annotated output video")
    args = parser.parse_args()

    run_on_video(args.source, args.config, args.save)