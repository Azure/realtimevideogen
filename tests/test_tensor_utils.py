#!/usr/bin/env python3

import os
import pytest
import binascii

import torch

from media_utils import tensor_to_base64
from file_utils import binary_to_base64
from file_utils import base64_to_binary
from media_utils import base64_to_tensor
from media_utils import bytes_to_tensor
from media_utils import get_tensor_file_info


def test_base64() -> None:
    """Test tensor to base64 and back conversion."""
    tensor_data = torch.rand(3, 4, 5)
    tensor_base64 = tensor_to_base64(tensor_data)
    assert isinstance(tensor_base64, str)

    tensor_data_2 = base64_to_tensor(tensor_base64)
    assert isinstance(tensor_data_2, torch.Tensor)
    assert tensor_data.shape == tensor_data_2.shape

    tensor_binary = base64_to_binary(tensor_base64)
    assert isinstance(tensor_binary, bytes)

    tensor_base64 = binary_to_base64(tensor_binary)
    assert isinstance(tensor_base64, str)

    tensor_data_3 = base64_to_tensor(tensor_base64)
    assert isinstance(tensor_data_3, torch.Tensor)
    assert tensor_data.shape == tensor_data_3.shape


def test_tensor_file() -> None:
    """Test saving tensor to file and getting its info."""
    tensor_data = torch.rand(3, 4, 5)
    tensor_base64 = tensor_to_base64(tensor_data)
    tensor_binary = base64_to_binary(tensor_base64)

    with open("test_tensor.pt", "wb") as file:
        file.write(tensor_binary)

    tensor_info = get_tensor_file_info("test_tensor.pt")
    assert tensor_info["dtype"].startswith("torch.float")
    assert tensor_info["shape"] == "torch.Size([3, 4, 5])"
    assert tensor_info["numel"] == 60

    os.remove("test_tensor.pt")

    with pytest.raises(TypeError):
        get_tensor_file_info(None)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        get_tensor_file_info("nonexisting.pt")


def test_base64_invalid() -> None:
    """Test invalid inputs for base64 and tensor functions."""
    with pytest.raises(TypeError):
        base64_to_binary(b"12345")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        base64_to_tensor(12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tensor_to_base64("12345")
    with pytest.raises(binascii.Error):
        base64_to_tensor("NOTBASE64")
    with pytest.raises(TypeError):
        bytes_to_tensor("12345")  # type: ignore[arg-type]
