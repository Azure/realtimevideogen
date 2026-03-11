"""
Utility functions for console applications.
"""
import os
import sys
import logging

from typing import Union
from typing import List

from colorlog import ColoredFormatter


def setup_logging(
    path: str = "/tmp",
    file_name: Union[str, List[str]] = "file.log",
    level: int = logging.INFO,
    use_global: bool = True,
) -> logging.Logger:
    """Setup logging configuration.
    Output logs to a file and stdout."""

    if isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = file_name

    if not os.path.exists(path):
        os.makedirs(path)

    logger_name = file_names[0]
    if use_global:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = ColoredFormatter(
        fmt="[%(asctime)s.%(msecs)03d] %(log_color)s%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            # "INFO": None,
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )

    for file_name in file_names:
        file_handler = logging.FileHandler(f"{path}/{file_name}")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


def bytes_to_human(num_bytes: int | float) -> str:
    """Convert a byte count to a human-readable string."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} EB"
