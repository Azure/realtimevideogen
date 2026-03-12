#!/usr/bin/env python3

import pytest

from file_utils import binary_to_base64
from file_utils import base64_to_binary


def test_binary_base64_conversion() -> None:
    """Test binary to base64 and back conversion."""
    original_binary = b"Hello, World!"
    base64_str = binary_to_base64(original_binary)
    converted_binary = base64_to_binary(base64_str)
    assert original_binary == converted_binary

    with pytest.raises(TypeError, match="Expected bytes for binary_data"):
        binary_to_base64("Not bytes")  # type: ignore
    with pytest.raises(TypeError, match="Expected str for base64_str"):
        base64_to_binary(12345)  # type: ignore
