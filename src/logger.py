"""Centralized logging configuration for the flight fare prediction pipeline."""

import logging
import os

def get_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Name of the logger (typically __name__ of the calling module).
        log_file: Path to the log file.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fh_format)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_format = logging.Formatter("%(levelname)s | %(message)s")
        ch.setFormatter(ch_format)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
