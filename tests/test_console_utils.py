#!/usr/bin/env python3

import logging
import pytest

from console_utils import bytes_to_human
from console_utils import setup_logging


def test_bytes_to_human() -> None:
    """Test the bytes_to_human function with various inputs."""
    assert bytes_to_human(0) == "0 B"
    assert bytes_to_human(512) == "512 B"
    assert bytes_to_human(1024) == "1.00 KB"
    assert bytes_to_human(1536) == "1.50 KB"
    assert bytes_to_human(1536000) == "1.46 MB"
    assert bytes_to_human(1.5 * 1024**3) == "1.50 GB"
    assert bytes_to_human(1.7 * 1024**4) == "1.70 TB"
    assert bytes_to_human(1.6 * 1024**5) == "1.60 PB"
    assert bytes_to_human(1.92 * 1024**6) == "1.92 EB"

    with pytest.raises(ValueError):
        bytes_to_human(-1)


def test_setup_logging() -> None:
    setup_logging(
        level=logging.DEBUG,
        path="/tmp",
        file_name="test_debug.log")

    setup_logging(
        level=logging.INFO,
        file_name="test_info.log")
