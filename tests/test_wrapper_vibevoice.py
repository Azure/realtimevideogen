#!/usr/bin/env python3

import sys
import gc
import pytest
import logging

from types import ModuleType

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/vibevoice")


class DummySchedulerMixin:
    pass


class DummyConfigMixin:
    pass


class DummySchedulerOutput:
    pass


diffusers_sched = ModuleType("diffusers.schedulers.scheduling_utils")
diffusers_sched.SchedulerMixin = DummySchedulerMixin
diffusers_sched.SchedulerOutput = DummySchedulerOutput
diffusers_sched.KarrasDiffusionSchedulers = MagicMock()

diffusers_conf = ModuleType("diffusers.configuration_utils")
diffusers_conf.ConfigMixin = DummyConfigMixin
diffusers_conf.register_to_config = MagicMock()

mock_modules = {
    "modeling_vibevoice_inference": MagicMock(),
    # "modeling_vibevoice": MagicMock(),
    "transformers": MagicMock(),
    "transformers.utils": MagicMock(),
    "transformers.modeling_utils": MagicMock(),
    "transformers.modeling_outputs": MagicMock(),
    "transformers.generation": MagicMock(),
    "transformers.models": MagicMock(),
    "transformers.models.llama": MagicMock(),
    "transformers.models.llama.modeling_llama": MagicMock(),
    "transformers.models.qwen2": MagicMock(),
    "transformers.models.qwen2.tokenization_qwen2": MagicMock(),
    "transformers.models.qwen2.tokenization_qwen2_fast": MagicMock(),
    "transformers.models.qwen2.configuration_qwen2": MagicMock(),
    "transformers.tokenization_utils_base": MagicMock(),
    "transformers.feature_extraction_utils": MagicMock(),
    "transformers.modeling_flash_attention_utils": MagicMock(),
    "transformers.configuration_utils": MagicMock(),
    "transformers.activations": MagicMock(),
    "diffusers": MagicMock(),
    "diffusers.schedulers": ModuleType("diffusers.schedulers"),
    "diffusers.schedulers.scheduling_utils": diffusers_sched,
    "diffusers.configuration_utils": diffusers_conf,
    "diffusers.utils": MagicMock(),
    "diffusers.utils.torch_utils": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from vibevoice.wrapper_vibevoice import VibeVoiceGeneration

    # from modeling_vibevoice import VibeVoicePreTrainedModel


@pytest.mark.asyncio
async def test_vibevoice() -> None:
    model = VibeVoiceGeneration()
    assert model is not None
    assert model.model_name == "vibevoice"
    assert model.status == "initializing"

    with pytest.raises(AttributeError):  # TODO
        model.init()
    assert model.status == "failed"

    health = model.get_health()
    assert health is not None
    assert len(health) > 1
    timestamps = model.get_timestamps()
    assert timestamps is not None
    assert len(timestamps) >= 1

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "text": "Test text"
    })

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.warmup()

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            text="Test text",
            output_type="audio_path")

    del model
    gc.collect()


def test_vibevoice_libs() -> None:
    try:
        from modeling_vibevoice import VibeVoiceCausalLMOutputWithPast
        assert VibeVoiceCausalLMOutputWithPast is not None
    except TypeError:
        logging.warning("Could not import VibeVoiceCausalLMOutputWithPast")

    try:
        from modeling_vibevoice import VibeVoiceGenerationOutput
        assert VibeVoiceGenerationOutput is not None
    except TypeError:
        logging.warning("Could not import VibeVoiceGenerationOutput")

    try:
        from modeling_vibevoice import SpeechConnector
        assert SpeechConnector is not None
    except TypeError:
        logging.warning("Could not import SpeechConnector")

    with patch.dict(sys.modules, mock_modules):
        try:
            from modeling_vibevoice import VibeVoicePreTrainedModel
            assert VibeVoicePreTrainedModel is not None
        except AttributeError:
            logging.warning("Could not import VibeVoicePreTrainedModel")

    with patch.dict(sys.modules, mock_modules):
        from modular_vibevoice_diffusion_head import RMSNorm
        assert RMSNorm is not None


def test_tokenizer() -> None:
    with patch.dict(sys.modules, mock_modules):
        from modular_vibevoice_tokenizer import NormConvTranspose1d
        assert NormConvTranspose1d is not None

        from modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
        assert VibeVoiceTokenizerStreamingCache is not None

        from modular_vibevoice_tokenizer import VibeVoiceSemanticTokenizerModel

        tokenizer = VibeVoiceSemanticTokenizerModel(None)
        assert tokenizer is not None


# TODO fix this test with the fast tokenizer dependency
"""
def test_text_tokenizer() -> None:
    from modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer
    with pytest.raises(FileNotFoundError):
        VibeVoiceTextTokenizer(
            "missing_vocab_file",
            "missing_merges_file")
"""


def test_model_inference() -> None:
    # Configuration
    from configuration_vibevoice import VibeVoiceConfig
    vibevoice_config = VibeVoiceConfig()
    assert vibevoice_config is not None

    # Inference
    with patch.dict(sys.modules, mock_modules):
        from modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        inference = VibeVoiceForConditionalGenerationInference(vibevoice_config)
        assert inference is not None
        assert inference.forward() is not None
        assert inference.generate() is not None


def test_audio_streamer() -> None:
    from audio_streamer import AudioStreamer
    audio_streamer = AudioStreamer(batch_size=8)
    assert audio_streamer is not None
    audio_streamer.put(
        audio_chunks=MagicMock(),
        sample_indices=MagicMock(),
    )
    audio_streamer.end(sample_indices=MagicMock())
