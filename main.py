# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Defense Surveillance System — Full Pipeline
#
# Pipeline:
#   Video → Detection → Tracking → Prediction → Threat Analysis → Output
#
# Usage:
#   python main.py --source data/raw/test.mp4 --save
#   python main.py --source data/raw/test.mp4 --draw-zones   # draw zones first
#   python main.py --source data/raw/test.mp4 --no-predict   # skip prediction
#   python main.py --source 0                                 # webcam
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import yaml
import argparse
import sys
import time
from pathlib import Path

from src.utils.logger import setup_logger, logger
from src.detection.detector import ObjectDetector
from src.tracking.tracker import ObjectTracker
from src.prediction.predictor import TrajectoryPredictor
from src.threat.threat_analyzer import ThreatAnalyzer, draw_zones, draw_alerts
from src.threat.zone_drawer import draw_zones_on_video


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self.times = []
        self.smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        self.times.append(now)
        if len(self.times) > self.smoothing:
            self.times.pop(0)
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame, frame_id, fps, n_det, n_tracks, n_pred, n_alerts, model_loaded):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "DEFENSE SURVEILLANCE SYSTEM  |  Detection + Tracking + Prediction + Threat Analysis",
                (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)

    model_status = "Model:TRAINED" if model_loaded else "Model:UNTRAINED"
    alert_status = f"ALERTS:{n_alerts}" if n_alerts == 0 else f"ALERTS:{n_alerts} !!!"
    alert_color  = (0, 255, 100) if n_alerts == 0 else (0, 0, 255)

    cv2.putText(frame,
                f"Frame:{frame_id:05d}  FPS:{fps:.1f}  Det:{n_det}  "
                f"Tracks:{n_tracks}  Pred:{n_pred}  {model_status}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

    cv2.putText(frame, alert_status, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1)

    # Legend bottom left
    cv2.putText(frame, "Solid=observed  Dashed=predicted  Red zone=restricted",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    # Status dot — red if alerts, green if clear
    dot_color = (0, 0, 255) if n_alerts > 0 else (0, 255, 80)
    cv2.circle(frame, (w - 18, 25), 8, dot_color, -1)
    return frame


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    source: str,
    config_path:    str  = "configs/config.yaml",
    zones_path:     str  = "configs/zones.json",
    save_output:    bool = False,
    show_window:    bool = True,
    use_prediction: bool = True,
    draw_zones_first: bool = False,
):
    config = load_config(config_path)
    setup_logger(
        log_file=config["logging"]["log_file"],
        level=config["logging"]["level"]
    )

    logger.info("=" * 60)
    logger.info("  Defense Surveillance System — Starting")
    logger.info("=" * 60)
    logger.info(f"Source     : {source}")
    logger.info(f"Prediction : {'enabled' if use_prediction else 'disabled'}")
    logger.info(f"Zones file : {zones_path}")

    # ── Step 0: Draw zones if requested ──────────────────────────────────────
    if draw_zones_first:
        logger.info("Opening zone drawer...")
        draw_zones_on_video(source, zones_path)
        logger.info("Zone drawing done. Starting pipeline...")

    # ── Initialize modules ────────────────────────────────────────────────────
    logger.info("Loading detector...")
    detector = ObjectDetector(config_path)

    logger.info("Loading tracker...")
    tracker = ObjectTracker(config_path)

    predictor    = None
    model_loaded = False
    if use_prediction:
        logger.info("Loading predictor...")
        predictor = TrajectoryPredictor(config_path)
        ckpt = Path(config["paths"]["models"]) / "predictor" / "checkpoint_best.pt"
        model_loaded = ckpt.exists()

    logger.info("Loading threat analyzer...")
    analyzer = ThreatAnalyzer(config_path, zones_path)

    # ── Open video ────────────────────────────────────────────────────────────
    src = int(source) if source == "0" else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        sys.exit(1)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video : {width}x{height} @ {video_fps:.1f} fps | {total_frames} frames")

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if save_output:
        out_path = Path("outputs/visualizations/pipeline_output.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
        )
        logger.info(f"Saving → {out_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    fps_counter  = FPSCounter()
    frame_id     = 0
    total_alerts = 0
    logger.info("Running... Press 'q' to quit, 's' to snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video.")
            break

        # 1. Detect
        detections = detector.detect(frame, frame_id)

        # 2. Track
        tracks = tracker.update(detections, frame, frame_id)

        # 3. Predict
        predictions = {}
        if predictor and tracks:
            predictions = predictor.predict_tracks(tracks)

        # 4. Threat analysis
        alerts = analyzer.analyze(tracks, predictions, frame_id, video_fps)
        total_alerts += len(alerts)

        # 5. Draw everything
        viz = detector.draw(frame, detections)
        viz = tracker.draw(viz, tracks)
        if predictor:
            viz = predictor.draw_predictions(viz, predictions, tracks)

        # Draw restricted zones
        viz = draw_zones(viz, analyzer.zones)

        # Draw alerts
        viz = draw_alerts(viz, alerts)

        # HUD
        fps = fps_counter.tick()
        viz = draw_hud(
            viz, frame_id, fps,
            len(detections), len(tracks), len(predictions),
            len(alerts), model_loaded
        )

        # 6. Output
        if writer:
            writer.write(viz)

        if show_window:
            cv2.imshow("Defense Surveillance System", viz)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Stopped by user.")
                break
            elif key == ord("s"):
                snap = f"outputs/visualizations/snapshot_{frame_id:05d}.jpg"
                cv2.imwrite(snap, viz)
                logger.info(f"Snapshot → {snap}")

        if frame_id % 100 == 0 and frame_id > 0:
            logger.info(
                f"Frame {frame_id} | FPS {fps:.1f} | "
                f"Tracks {len(tracks)} | Alerts this frame {len(alerts)}"
            )

        frame_id += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Save alert log
    analyzer.save_alert_log("outputs/alerts/alert_log.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_traj = tracker.get_all_trajectories()
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete")
    logger.info(f"  Frames processed : {frame_id}")
    logger.info(f"  Total tracks     : {len(all_traj)}")
    logger.info(f"  Total alerts     : {total_alerts}")
    logger.info(f"  Alert log        : outputs/alerts/alert_log.json")
    logger.info("=" * 60)

    return all_traj


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defense Surveillance System")
    parser.add_argument("--source",      type=str, default="0")
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--zones",       type=str, default="configs/zones.json")
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--no-display",  action="store_true")
    parser.add_argument("--no-predict",  action="store_true")
    parser.add_argument("--draw-zones",  action="store_true",
                        help="Draw restricted zones before running pipeline")
    args = parser.parse_args()

    run_pipeline(
        source          = args.source,
        config_path     = args.config,
        zones_path      = args.zones,
        save_output     = args.save,
        show_window     = not args.no_display,
        use_prediction  = not args.no_predict,
        draw_zones_first= args.draw_zones,
    )