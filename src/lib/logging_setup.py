"""Uniform logging. Every stage logs to console and to outputs/<run_id>/logs/."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(stage: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(stage)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{stage}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
