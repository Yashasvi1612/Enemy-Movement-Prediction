# src/detection/detector.py
# Phase 1 — Object Detection using YOLOv8

import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from loguru import logger


@dataclass
class Detection:
    """Single object detection result from one frame."""
    bbox: Tuple[int, int, int, int]
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


class ObjectDetector:
    CLASS_NAMES = {
        0: "person",
        2: "car",
        3: "motorbike",
        5: "bus",
        7: "truck"
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config  = self._load_config(config_path)
        self.det_cfg = self.config["detection"]
        self.model   = self._load_model()
        logger.info(f"Detector initialized | model: {self.det_cfg['model_size']} | device: {self.det_cfg['device']}")

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            return YOLO(self.det_cfg["model_size"])
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
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
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
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