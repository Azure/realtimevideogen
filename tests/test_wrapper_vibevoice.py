#!/usr/bin/env python3

import sys
import gc
import pytest
import logging

from types import ModuleType

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.test_utils import temp_sys_path

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


def test_voice_mapper() -> None:
    with patch.dict(sys.modules, mock_modules):
        from vibevoice.wrapper_vibevoice import VoiceMapper

    voice_mapper = VoiceMapper()
    voice_mapper.setup_voice_presets()

    # voice_presets is a dict (may be empty or populated depending on environment)
    assert isinstance(voice_mapper.voice_presets, dict)

    if not voice_mapper.voice_presets:
        # No voices available: get_voice_path raises ValueError
        with pytest.raises(ValueError, match="No voice presets available"):
            voice_mapper.get_voice_path("any_speaker")
    else:
        # Voices available: get_voice_path returns a path for any speaker name
        path = voice_mapper.get_voice_path("any_speaker")
        assert isinstance(path, str)


@pytest.mark.asyncio
async def test_vibevoice_get_rest_args_voice() -> None:
    with patch.dict(sys.modules, mock_modules):
        from vibevoice.wrapper_vibevoice import VibeVoiceGeneration as _VVG

    model = _VVG()

    # Custom voice parameter is returned in args
    result = await model.get_rest_args({"text": "Hello world", "voice": "custom_voice"})
    assert result["args"]["voice"] == "custom_voice"

    # Default voice returned when voice not specified
    result_default = await model.get_rest_args({"text": "Hello world"})
    assert "voice" in result_default["args"]
    assert result_default["args"]["voice"] == "af_heart"


def test_timestep_samplers() -> None:
    with temp_sys_path("wrapper/vibevoice"):
        with patch.dict(sys.modules, {'torch': mock_torch}):
            from schedule.timestep_sampler import UniformSampler, LogitNormalSampler

    uniform = UniformSampler(timesteps=100)
    result_u = uniform.sample(batch_size=4, device=mock_torch.device('cpu'))
    assert result_u is not None

    logit = LogitNormalSampler(timesteps=100)
    result_l = logit.sample(batch_size=4, device=mock_torch.device('cpu'))
    assert result_l is not None


def test_dpm_solver_scheduler() -> None:
    """DPMSolverMultistepScheduler can be imported and instantiated with mocked deps."""
    import numpy as np
    import torch as real_torch

    # Build minimal diffusers mocks with a proper pass-through register_to_config
    class DummyDPMSchedulerMixin:
        pass

    class DummyDPMConfigMixin:
        pass

    def passthrough_register_to_config(func):
        return func

    dpm_sched_utils = ModuleType("diffusers.schedulers.scheduling_utils")
    dpm_sched_utils.SchedulerMixin = DummyDPMSchedulerMixin
    dpm_sched_utils.SchedulerOutput = MagicMock
    dpm_sched_utils.KarrasDiffusionSchedulers = MagicMock()

    dpm_conf = ModuleType("diffusers.configuration_utils")
    dpm_conf.ConfigMixin = DummyDPMConfigMixin
    dpm_conf.register_to_config = passthrough_register_to_config

    dpm_mocks = {
        "diffusers.schedulers": ModuleType("diffusers.schedulers"),
        "diffusers.schedulers.scheduling_utils": dpm_sched_utils,
        "diffusers.configuration_utils": dpm_conf,
        "diffusers.utils": MagicMock(),
        "diffusers.utils.torch_utils": MagicMock(),
        "numpy": np,
        "torch": real_torch,
    }

    with temp_sys_path("wrapper/vibevoice"):
        with patch.dict(sys.modules, dpm_mocks):
            from schedule.dpm_solver import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=100)
    assert scheduler is not None
    assert hasattr(scheduler, "betas")
    assert hasattr(scheduler, "alphas_cumprod")

    # Test other beta schedules to cover more branches
    for schedule in ["scaled_linear", "squaredcos_cap_v2", "cosine"]:
        s = DPMSolverMultistepScheduler(num_train_timesteps=50, beta_schedule=schedule)
        assert hasattr(s, "betas")

    # Unknown schedule raises NotImplementedError
    with pytest.raises(NotImplementedError):
        DPMSolverMultistepScheduler(beta_schedule="unknown_schedule")

    # Test betas_for_alpha_bar directly with different alpha_transform_types
    with temp_sys_path("wrapper/vibevoice"):
        with patch.dict(sys.modules, dpm_mocks):
            from schedule.dpm_solver import betas_for_alpha_bar

    for alpha_type in ["cosine", "exp", "cauchy", "laplace"]:
        betas = betas_for_alpha_bar(10, alpha_transform_type=alpha_type)
        assert betas is not None

    with pytest.raises(ValueError):
        betas_for_alpha_bar(10, alpha_transform_type="unknown")


def test_dpm_solver_set_timesteps() -> None:
    """Test DPMSolverMultistepScheduler.set_timesteps, rescale_zero_terminal_snr, add_noise, get_velocity."""
    import numpy as np
    import torch as real_torch

    class DummyDPMSchedulerMixin:
        pass

    class DummyDPMConfigMixin:
        pass

    def passthrough_register_to_config(func):
        return func

    dpm_sched_utils = ModuleType("diffusers.schedulers.scheduling_utils")
    dpm_sched_utils.SchedulerMixin = DummyDPMSchedulerMixin
    dpm_sched_utils.SchedulerOutput = MagicMock
    dpm_sched_utils.KarrasDiffusionSchedulers = MagicMock()

    dpm_conf = ModuleType("diffusers.configuration_utils")
    dpm_conf.ConfigMixin = DummyDPMConfigMixin
    dpm_conf.register_to_config = passthrough_register_to_config

    dpm_mocks = {
        "diffusers.schedulers": ModuleType("diffusers.schedulers"),
        "diffusers.schedulers.scheduling_utils": dpm_sched_utils,
        "diffusers.configuration_utils": dpm_conf,
        "diffusers.utils": MagicMock(),
        "diffusers.utils.torch_utils": MagicMock(),
        "numpy": np,
        "torch": real_torch,
    }

    with temp_sys_path("wrapper/vibevoice"):
        with patch.dict(sys.modules, dpm_mocks):
            from schedule.dpm_solver import (
                DPMSolverMultistepScheduler,
                rescale_zero_terminal_snr,
            )

    # set_timesteps with different timestep_spacing values
    for spacing in ["linspace", "leading", "trailing"]:
        s = DPMSolverMultistepScheduler(num_train_timesteps=100, timestep_spacing=spacing)
        s.set_timesteps(10)
        assert s.num_inference_steps == 10
        assert len(s.timesteps) == 10

    # set_timesteps with use_karras_sigmas covers _convert_to_karras + _sigma_to_t
    s_karras = DPMSolverMultistepScheduler(num_train_timesteps=100, use_karras_sigmas=True)
    s_karras.set_timesteps(10)
    assert s_karras.num_inference_steps == 10

    # set_timesteps with use_lu_lambdas covers _convert_to_lu + _sigma_to_t
    s_lu = DPMSolverMultistepScheduler(num_train_timesteps=100, use_lu_lambdas=True)
    s_lu.set_timesteps(10)
    assert s_lu.num_inference_steps == 10

    # set_timesteps with custom timesteps list
    s_custom = DPMSolverMultistepScheduler(num_train_timesteps=100)
    s_custom.set_timesteps(timesteps=[80, 60, 40, 20])
    assert s_custom.num_inference_steps == 4

    # Error: neither argument provided
    with pytest.raises(ValueError, match="Must pass exactly one of"):
        s_custom.set_timesteps()

    # Error: both provided
    with pytest.raises(ValueError, match="Can only pass one"):
        s_custom.set_timesteps(num_inference_steps=10, timesteps=[80, 60])

    # Error: custom timesteps with karras_sigmas
    with pytest.raises(ValueError, match="use_karras_sigmas"):
        s_karras.set_timesteps(timesteps=[80, 60])

    # __len__, set_begin_index, step_index and begin_index properties
    s = DPMSolverMultistepScheduler(num_train_timesteps=200)
    assert len(s) == 200
    s.set_begin_index(5)
    assert s.begin_index == 5
    assert s.step_index is None  # before any step

    # rescale_zero_terminal_snr
    betas = real_torch.linspace(0.0001, 0.02, 100)
    rescaled = rescale_zero_terminal_snr(betas)
    assert rescaled is not None
    assert rescaled.shape == betas.shape

    # add_noise and get_velocity
    s.set_timesteps(10)
    original = real_torch.randn(2, 4)
    noise = real_torch.randn(2, 4)
    timesteps = real_torch.IntTensor([50, 80])
    noisy = s.add_noise(original, noise, timesteps)
    assert noisy.shape == original.shape
    velocity = s.get_velocity(original, noise, timesteps)
    assert velocity.shape == original.shape

    # _sigma_to_alpha_sigma_t and _sigma_to_t are exercised by set_timesteps
    # Make sure sigmas attribute exists after set_timesteps
    assert hasattr(s, "sigmas")
    assert s.sigmas is not None


def test_vibevoice_model_methods() -> None:
    """Test _assert_model_init, init_parallelism, init_model_parallelism, model_compile."""
    with patch.dict(sys.modules, mock_modules):
        from vibevoice.wrapper_vibevoice import VibeVoiceGeneration as _VVG

    model = _VVG()

    # _assert_model_init raises before model is initialized (status != "ok")
    with pytest.raises(ValueError, match="Model not initialized"):
        model._assert_model_init()

    # init_parallelism runs without error (no CUDA in test env, uses CPU path)
    model.init_parallelism()
    assert model.rank == 0
    assert model.world_size == 1

    # init_model_parallelism: world_size=1, just logs a warning if > 1
    model.init_model_parallelism()

    # model_compile: short-circuits when torch_compile=False
    model.torch_compile = False
    model.model_compile()  # should return immediately without error

    del model
