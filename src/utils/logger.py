# src/utils/logger.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized logger for the entire project.
# Import this in any module instead of setting up loguru separately.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_file: str = "outputs/logs/system.log", level: str = "INFO"):
    """
    Configure loguru logger with console + file output.
    Call once at startup (in main.py).
    """
    # Remove default handler
    logger.remove()

    # Console — colored, human-readable
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File — full detail
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logger ready | level={level} | file={log_file}")
    return logger


# Re-export so any module can do: from src.utils.logger import logger
__all__ = ["logger", "setup_logger"]