# src/prediction/predictor.py
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger

from src.prediction.transformer_model import build_model, TrajectoryTransformer
from src.tracking.tracker import Track


class TrajectoryPredictor:
    """
    Loads trained Transformer and runs predictions on live tracks.

    Usage:
        predictor   = TrajectoryPredictor("configs/config.yaml")
        predictions = predictor.predict_tracks(active_tracks)
        # {track_id: [(x,y), ... x12 future points]}
    """

    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.obs_len  = self.config["trajectory"]["obs_len"]
        self.pred_len = self.config["trajectory"]["pred_len"]
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = self._load_model()

    def _load_model(self) -> TrajectoryTransformer:
        model     = build_model(self.config).to(self.device)
        model.eval()
        ckpt_path = Path(self.config["paths"]["models"]) / "predictor" / "checkpoint_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded model | epoch={ckpt.get('epoch','?')} | ADE={ckpt.get('val_ade',0):.4f}")
        else:
            logger.warning("No trained model found. Run: python -m src.prediction.trainer --data data/raw/eth_ucy/")
        return model

    def _normalize(self, positions):
        arr    = np.array(positions[-self.obs_len:], dtype=np.float32)
        origin = arr[-1]
        arr    = arr - origin
        scale  = np.abs(arr).max() + 1e-8
        return arr / scale, origin, scale

    def _denormalize(self, pred, origin, scale):
        pred = pred * scale + origin
        return [(float(x), float(y)) for x, y in pred]

    def predict_tracks(self, tracks: List[Track]) -> Dict[int, List[Tuple[float, float]]]:
        predictions  = {}
        eligible     = [t for t in tracks if t.trajectory_length >= self.obs_len]
        if not eligible:
            return predictions

        arrays, origins, scales, ids = [], [], [], []
        for t in eligible:
            arr, origin, scale = self._normalize(t.positions)
            arrays.append(arr); origins.append(origin)
            scales.append(scale); ids.append(t.track_id)

        obs_tensor = torch.tensor(np.stack(arrays), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model.predict(obs_tensor)
        pred_np = pred_tensor.cpu().numpy()

        for i, tid in enumerate(ids):
            predictions[tid] = self._denormalize(pred_np[i], origins[i], scales[i])

        return predictions

    def draw_predictions(self, frame, predictions, tracks):
        frame  = frame.copy()
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]
        for tid, pts in predictions.items():
            if len(pts) < 2:
                continue
            color      = colors[tid % len(colors)]
            pred_color = tuple(min(255, int(c * 1.3)) for c in color)

            for i in range(1, len(pts)):
                if i % 2 == 0:
                    pt1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                    pt2 = (int(pts[i][0]),   int(pts[i][1]))
                    cv2.line(frame, pt1, pt2, pred_color, 2)

            end = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.circle(frame, end, 6, pred_color, -1)
            cv2.circle(frame, end, 8, (255, 255, 255), 1)
            cv2.putText(frame, f"ID:{tid}", (end[0]+5, end[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1)
        return frame