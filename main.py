# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Defense Surveillance System — Main Pipeline Runner
#
# Runs the full pipeline on a video:
#   Video → Detection → Tracking → [Prediction → Threat] (coming in Phase 3+)
#
# Usage:
#   python main.py --source data/raw/sample.mp4
#   python main.py --source 0                      # webcam
#   python main.py --source data/raw/sample.mp4 --save
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


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
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
def draw_hud(
    frame,
    frame_id: int,
    fps: float,
    n_detections: int,
    n_tracks: int,
    phase: str = "Phase 1-2: Detection + Tracking"
):
    """Draw system info overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text
    cv2.putText(frame, f"DEFENSE SURVEILLANCE SYSTEM  |  {phase}",
                (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    cv2.putText(frame,
                f"Frame: {frame_id:05d}  |  FPS: {fps:.1f}  |  "
                f"Detections: {n_detections}  |  Tracks: {n_tracks}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

    # Status dot (green = running)
    cv2.circle(frame, (w - 18, 20), 7, (0, 255, 80), -1)
    return frame


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    source: str,
    config_path: str = "configs/config.yaml",
    save_output: bool = False,
    show_window: bool = True,
):
    config = load_config(config_path)
    setup_logger(
        log_file=config["logging"]["log_file"],
        level=config["logging"]["level"]
    )

    logger.info("=" * 60)
    logger.info("  Defense Surveillance System — Starting")
    logger.info("=" * 60)
    logger.info(f"Source : {source}")
    logger.info(f"Config : {config_path}")

    # ── Initialize modules ────────────────────────────────────────────────────
    logger.info("Loading detector...")
    detector = ObjectDetector(config_path)

    logger.info("Loading tracker...")
    tracker = ObjectTracker(config_path)

    # ── Open video source ─────────────────────────────────────────────────────
    src = int(source) if source == "0" else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video  : {width}x{height} @ {video_fps:.1f} fps | {total_frames} frames")

    # ── Output video writer ───────────────────────────────────────────────────
    writer = None
    if save_output:
        out_path = Path("outputs/visualizations/pipeline_output.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (width, height))
        logger.info(f"Saving output → {out_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    fps_counter = FPSCounter()
    frame_id    = 0

    logger.info("Pipeline running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        # 1. Detect objects
        detections = detector.detect(frame, frame_id)

        # 2. Track objects
        tracks = tracker.update(detections, frame, frame_id)

        # 3. Draw detections (boxes + labels)
        viz = detector.draw(frame, detections)

        # 4. Draw tracks (IDs + trails)
        viz = tracker.draw(viz, tracks)

        # 5. HUD overlay
        fps = fps_counter.tick()
        viz = draw_hud(viz, frame_id, fps, len(detections), len(tracks))

        # 6. Save / display
        if writer:
            writer.write(viz)

        if show_window:
            cv2.imshow("Defense Surveillance System", viz)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Stopped by user.")
                break
            elif key == ord("s"):
                snap_path = f"outputs/visualizations/snapshot_{frame_id:05d}.jpg"
                cv2.imwrite(snap_path, viz)
                logger.info(f"Snapshot saved → {snap_path}")

        # Log every 100 frames
        if frame_id % 100 == 0 and frame_id > 0:
            logger.info(f"Frame {frame_id} | FPS {fps:.1f} | "
                        f"Detections {len(detections)} | Active tracks {len(tracks)}")

        frame_id += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_trajectories = tracker.get_all_trajectories()
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete")
    logger.info(f"  Frames processed : {frame_id}")
    logger.info(f"  Total tracks     : {len(all_trajectories)}")
    logger.info(f"  Tracks with 8+ points (usable for prediction): "
                f"{sum(1 for t in all_trajectories.values() if len(t) >= 8)}")
    logger.info("=" * 60)

    return tracker.get_all_trajectories()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Defense Surveillance System — Pipeline Runner"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video file path or '0' for webcam"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output video"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no window, useful for servers)"
    )
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        config_path=args.config,
        save_output=args.save,
        show_window=not args.no_display,
    )