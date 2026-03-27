#!/usr/bin/env python3
"""
Tests for wrapper/hunyuanavatar/encode_data.py
"""

import sys
import numpy as np

from unittest.mock import patch, MagicMock
from tests.torch_mock import TorchMock
from PIL import Image

mock_torch = TorchMock()

mock_modules = {
    'nvidia_smi': MagicMock(),
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'transformers': MagicMock(),
    'einops': MagicMock(),
    'librosa': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanavatar")

with patch.dict(sys.modules, mock_modules):
    from hunyuanavatar.encode_data import get_audio_feature
    from hunyuanavatar.encode_data import VideoAudioTextLoaderVal
    _encode_data_module = sys.modules['hunyuanavatar.encode_data']


def test_get_audio_feature_basic() -> None:
    """Test get_audio_feature returns expected tuple structure."""
    mock_feature_extractor = MagicMock()
    fake_input_features = MagicMock()
    fake_input_features.__len__ = MagicMock(return_value=1)
    mock_feature_extractor.return_value.input_features = fake_input_features

    mock_audio_input = np.zeros(16000 * 2, dtype=np.float32)  # 2s @ 16kHz

    with patch.object(_encode_data_module, 'librosa') as mock_librosa, \
         patch.object(_encode_data_module, 'torch') as mock_t:

        mock_librosa.load.return_value = (mock_audio_input, 16000)

        fake_tensor = MagicMock()
        mock_t.cat.return_value = fake_tensor

        result_features, result_len = get_audio_feature(mock_feature_extractor, "/tmp/test.wav")

    assert result_features is fake_tensor
    assert result_len == len(mock_audio_input) // 640


def test_video_audio_text_loader_val_init() -> None:
    """Test VideoAudioTextLoaderVal initializes correctly."""
    mock_text_encoder = MagicMock()
    mock_text_encoder_2 = MagicMock()
    mock_feature_extractor = MagicMock()

    with patch.object(_encode_data_module, 'torch') as mock_t, \
         patch.object(_encode_data_module, 'transforms'), \
         patch.object(_encode_data_module, 'CLIPImageProcessor'):

        mock_t.device.return_value = MagicMock()
        mock_t.float16 = mock_torch.float16

        loader = VideoAudioTextLoaderVal(
            image_size=704,
            text_encoder=mock_text_encoder,
            text_encoder_2=mock_text_encoder_2,
            feature_extractor=mock_feature_extractor,
        )

    assert loader.image_size == 704
    assert loader.text_encoder is mock_text_encoder
    assert loader.text_encoder_2 is mock_text_encoder_2
    assert loader.feature_extractor is mock_feature_extractor


def test_video_audio_text_loader_val_get_text_tokens() -> None:
    """Test VideoAudioTextLoaderVal.get_text_tokens static method."""
    mock_text_encoder = MagicMock()
    fake_input_ids = MagicMock()
    fake_attn_mask = MagicMock()
    fake_input_ids.squeeze.return_value = fake_input_ids
    fake_attn_mask.squeeze.return_value = fake_attn_mask
    mock_text_encoder.text2tokens.return_value = {
        "input_ids": fake_input_ids,
        "attention_mask": fake_attn_mask,
    }

    text_ids, text_mask = VideoAudioTextLoaderVal.get_text_tokens(
        mock_text_encoder, "test description"
    )
    mock_text_encoder.text2tokens.assert_called_once_with(
        "test description", data_type="video"
    )
    assert text_ids is fake_input_ids
    assert text_mask is fake_attn_mask


def test_video_audio_text_loader_val_get_text_tokens_image_dtype() -> None:
    """Test get_text_tokens with image dtype_encode."""
    mock_text_encoder = MagicMock()
    fake_input_ids = MagicMock()
    fake_attn_mask = MagicMock()
    fake_input_ids.squeeze.return_value = fake_input_ids
    fake_attn_mask.squeeze.return_value = fake_attn_mask
    mock_text_encoder.text2tokens.return_value = {
        "input_ids": fake_input_ids,
        "attention_mask": fake_attn_mask,
    }

    text_ids, text_mask = VideoAudioTextLoaderVal.get_text_tokens(
        mock_text_encoder, "a photo", dtype_encode="image"
    )
    mock_text_encoder.text2tokens.assert_called_once_with(
        "a photo", data_type="image"
    )


def test_video_audio_text_loader_encode_data_returns_expected_structure() -> None:
    """Test encode_data returns the expected dict structure with correct keys and values."""
    mock_text_encoder = MagicMock()
    mock_text_encoder_2 = MagicMock()
    mock_feature_extractor = MagicMock()

    # Prepare fake returns for text tokens
    def _fake_text2tokens(text: str, data_type: str = "video") -> dict:
        fake_ids = MagicMock()
        fake_ids.squeeze.return_value = MagicMock()
        fake_mask = MagicMock()
        fake_mask.squeeze.return_value = MagicMock()
        return {"input_ids": fake_ids, "attention_mask": fake_mask}

    mock_text_encoder.text2tokens.side_effect = _fake_text2tokens
    mock_text_encoder_2.text2tokens.side_effect = _fake_text2tokens

    ref_image = Image.new("RGB", (120, 80))

    # Set up a fake pixel_values tensor that supports the operations in encode_data
    fake_np_image = np.zeros((64, 64, 3), dtype=np.uint8)
    fake_item = MagicMock()
    fake_item.permute.return_value.data.cpu.return_value.numpy.return_value.astype.return_value = fake_np_image
    fake_pixel_values = MagicMock()
    fake_pixel_values.__getitem__ = MagicMock(return_value=fake_item)
    fake_pixel_values.__iter__ = MagicMock(return_value=iter([fake_item]))

    with patch.object(_encode_data_module, 'torch') as mock_t, \
         patch.object(_encode_data_module, 'transforms'), \
         patch.object(_encode_data_module, 'CLIPImageProcessor'), \
         patch.object(_encode_data_module, 'rearrange') as mock_rearrange, \
         patch.object(_encode_data_module, 'ToPILImage'), \
         patch.object(_encode_data_module, 'get_audio_feature') as mock_get_audio:

        mock_t.float16 = mock_torch.float16
        mock_t.from_numpy.side_effect = lambda x: MagicMock()
        mock_t.device.return_value = MagicMock()

        fake_audio_features = MagicMock()
        fake_audio_features.__getitem__ = MagicMock(return_value=MagicMock())
        mock_get_audio.return_value = (fake_audio_features, 10)

        # Make rearrange return our controlled fake_pixel_values
        mock_rearrange.return_value = fake_pixel_values

        # Create loader
        loader = VideoAudioTextLoaderVal(
            image_size=704,
            text_encoder=mock_text_encoder,
            text_encoder_2=mock_text_encoder_2,
            feature_extractor=mock_feature_extractor,
        )

        result = loader.encode_data(
            ref_image=ref_image,
            audio_path="/tmp/test_audio.wav",
            prompt="test prompt",
            fps=12.5,
        )

    assert result is not None
    assert "text_prompt" in result
    assert "audio_len" in result
    assert result["audio_len"] == 10
    assert result["audio_path"] == "/tmp/test_audio.wav"


def test_video_audio_text_loader_encode_data_large_image() -> None:
    """Test encode_data with a large image that triggers the rescaling branch (lines 95-98)."""
    mock_text_encoder = MagicMock()
    mock_text_encoder_2 = MagicMock()
    mock_feature_extractor = MagicMock()

    def _fake_text2tokens(text: str, data_type: str = "video") -> dict:
        fake_ids = MagicMock()
        fake_ids.squeeze.return_value = MagicMock()
        fake_mask = MagicMock()
        fake_mask.squeeze.return_value = MagicMock()
        return {"input_ids": fake_ids, "attention_mask": fake_mask}

    mock_text_encoder.text2tokens.side_effect = _fake_text2tokens
    mock_text_encoder_2.text2tokens.side_effect = _fake_text2tokens

    # Large image: 1000x2000 triggers the area-cap branch in encode_data.
    # At image_size=704: new_w*new_h exceeds the cap of 704*1216 (portrait max area),
    # so the code re-scales to fit within that budget.
    ref_image = Image.new("RGB", (1000, 2000))

    fake_np_image = np.zeros((64, 64, 3), dtype=np.uint8)
    fake_item = MagicMock()
    fake_item.permute.return_value.data.cpu.return_value.numpy.return_value.astype.return_value = fake_np_image
    fake_pixel_values = MagicMock()
    fake_pixel_values.__getitem__ = MagicMock(return_value=fake_item)
    fake_pixel_values.__iter__ = MagicMock(return_value=iter([fake_item]))

    with patch.object(_encode_data_module, 'torch') as mock_t, \
         patch.object(_encode_data_module, 'transforms'), \
         patch.object(_encode_data_module, 'CLIPImageProcessor'), \
         patch.object(_encode_data_module, 'rearrange') as mock_rearrange, \
         patch.object(_encode_data_module, 'ToPILImage'), \
         patch.object(_encode_data_module, 'get_audio_feature') as mock_get_audio:

        mock_t.float16 = mock_torch.float16
        mock_t.from_numpy.side_effect = lambda x: MagicMock()
        mock_t.device.return_value = MagicMock()
        fake_audio_features = MagicMock()
        fake_audio_features.__getitem__ = MagicMock(return_value=MagicMock())
        mock_get_audio.return_value = (fake_audio_features, 10)
        mock_rearrange.return_value = fake_pixel_values

        loader = VideoAudioTextLoaderVal(
            image_size=704,
            text_encoder=mock_text_encoder,
            text_encoder_2=mock_text_encoder_2,
            feature_extractor=mock_feature_extractor,
        )
        result = loader.encode_data(
            ref_image=ref_image,
            audio_path="/tmp/test_audio.wav",
            prompt="large image test",
            fps=12.5,
        )

    assert result is not None
    assert result["audio_len"] == 10
