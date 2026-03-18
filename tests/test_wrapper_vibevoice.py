#!/usr/bin/env python3

import os
import sys
import gc
import base64
import inspect
import tempfile
import pytest

import torch

from types import ModuleType

from unittest.mock import patch
from unittest.mock import MagicMock

sys.path.append("wrapper")
sys.path.append("wrapper/vibevoice")


class DummySchedulerMixin:
    pass


class _FrozenDict(dict):
    """Dict-like config that also supports attribute access (mimics diffusers FrozenDict)."""
    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class DummyConfigMixin:
    """Mimics diffusers ConfigMixin: register_to_config stores __init__ kwargs in self.config."""
    pass


class DummySchedulerOutput:
    pass


class DummyModelOutput:
    """Minimal stand-in for transformers.ModelOutput so @dataclass can resolve __mro__."""
    pass


class DummyPretrainedConfig:
    """Minimal stand-in for transformers.PretrainedConfig."""
    model_type = ""
    is_composition = False
    sub_configs = {}

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyBaseModelOutputWithPast(DummyModelOutput):
    pass


class DummyPreTrainedModel(torch.nn.Module):
    """Minimal stand-in for transformers.PreTrainedModel."""
    config_class = None
    base_model_prefix = ""

    def __init__(self, config: object = None, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: object) -> None:
        pass

    def post_init(self) -> None:
        pass


class DummyLlamaRMSNorm(torch.nn.Module):
    """Minimal stand-in for LlamaRMSNorm."""
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()

    def forward(self, x: object) -> object:
        return x


def passthrough_register_to_config(func):
    """Mimics @register_to_config: wraps __init__ to store kwargs as self.config."""
    sig = inspect.signature(func)

    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        cfg = {k: v for k, v in bound.arguments.items() if k != "self"}
        self.config = _FrozenDict(cfg)
        return func(self, *args, **kwargs)

    return wrapper


diffusers_sched = ModuleType("diffusers.schedulers.scheduling_utils")
diffusers_sched.SchedulerMixin = DummySchedulerMixin
diffusers_sched.SchedulerOutput = DummySchedulerOutput
diffusers_sched.KarrasDiffusionSchedulers = MagicMock()

diffusers_conf = ModuleType("diffusers.configuration_utils")
diffusers_conf.ConfigMixin = DummyConfigMixin
diffusers_conf.register_to_config = passthrough_register_to_config

# Build transformers mock modules with real classes where needed for inheritance
mock_transformers = MagicMock()

mock_modeling_outputs = MagicMock()
mock_modeling_outputs.ModelOutput = DummyModelOutput
mock_modeling_outputs.BaseModelOutputWithPast = DummyBaseModelOutputWithPast

mock_modeling_utils = MagicMock()
mock_modeling_utils.PreTrainedModel = DummyPreTrainedModel

mock_llama_modeling = MagicMock()
mock_llama_modeling.LlamaRMSNorm = DummyLlamaRMSNorm

mock_transformers_config = ModuleType("transformers.configuration_utils")
mock_transformers_config.PretrainedConfig = DummyPretrainedConfig

mock_modules = {
    "modeling_vibevoice_inference": MagicMock(),
    # "modeling_vibevoice": MagicMock(),
    "transformers": mock_transformers,
    "transformers.utils": MagicMock(),
    "transformers.modeling_utils": mock_modeling_utils,
    "transformers.modeling_outputs": mock_modeling_outputs,
    "transformers.generation": MagicMock(),
    "transformers.models": MagicMock(),
    "transformers.models.llama": MagicMock(),
    "transformers.models.llama.modeling_llama": mock_llama_modeling,
    "transformers.models.qwen2": MagicMock(),
    "transformers.models.qwen2.tokenization_qwen2": MagicMock(),
    "transformers.models.qwen2.tokenization_qwen2_fast": MagicMock(),
    "transformers.models.qwen2.configuration_qwen2": MagicMock(),
    "transformers.tokenization_utils_base": MagicMock(),
    "transformers.feature_extraction_utils": MagicMock(),
    "transformers.modeling_flash_attention_utils": MagicMock(),
    "transformers.configuration_utils": mock_transformers_config,
    "transformers.activations": MagicMock(),
    "diffusers": MagicMock(),
    "diffusers.schedulers": ModuleType("diffusers.schedulers"),
    "diffusers.schedulers.scheduling_utils": diffusers_sched,
    "diffusers.configuration_utils": diffusers_conf,
    "diffusers.utils": MagicMock(),
    "diffusers.utils.torch_utils": MagicMock(),
}


with patch.dict(sys.modules, mock_modules):
    from vibevoice.wrapper_vibevoice import VibeVoiceGeneration
    from vibevoice.wrapper_vibevoice import VoiceMapper

    from configuration_vibevoice import VibeVoiceConfig
    from configuration_vibevoice import VibeVoiceSemanticTokenizerConfig
    from modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from audio_streamer import AudioStreamer

    # from modeling_vibevoice import VibeVoicePreTrainedModel
    from modeling_vibevoice import VibeVoiceCausalLMOutputWithPast
    from modeling_vibevoice import SpeechConnector
    from modeling_vibevoice import VibeVoiceGenerationOutput
    from modeling_vibevoice import VibeVoicePreTrainedModel
    from modular_vibevoice_diffusion_head import RMSNorm
    from modular_vibevoice_tokenizer import NormConvTranspose1d
    from modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
    from modular_vibevoice_tokenizer import VibeVoiceSemanticTokenizerModel
    from schedule.timestep_sampler import UniformSampler, LogitNormalSampler
    from schedule.dpm_solver import DPMSolverMultistepScheduler
    from schedule.dpm_solver import rescale_zero_terminal_snr
    from schedule.dpm_solver import betas_for_alpha_bar
    from modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer


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
    assert VibeVoiceCausalLMOutputWithPast is not None
    assert VibeVoiceGenerationOutput is not None
    assert SpeechConnector is not None
    assert VibeVoicePreTrainedModel is not None
    assert RMSNorm is not None


def test_tokenizer() -> None:
    assert NormConvTranspose1d is not None
    assert VibeVoiceTokenizerStreamingCache is not None
    tokenizer = VibeVoiceSemanticTokenizerModel(VibeVoiceSemanticTokenizerConfig())
    assert tokenizer is not None


def test_text_tokenizer() -> None:
    tokenizer = VibeVoiceTextTokenizer(
        vocab_file="missing_vocab_file",
        merges_file="missing_merges_file")
    assert tokenizer is not None


def test_model_inference() -> None:
    # Configuration
    vibevoice_config = VibeVoiceConfig()
    assert vibevoice_config is not None

    # Inference
    inference = VibeVoiceForConditionalGenerationInference(vibevoice_config)
    assert inference is not None
    assert inference.forward() is not None
    assert inference.generate() is not None


def test_audio_streamer() -> None:
    audio_streamer = AudioStreamer(batch_size=8)
    assert audio_streamer is not None
    audio_streamer.put(
        audio_chunks=MagicMock(),
        sample_indices=MagicMock(),
    )
    audio_streamer.end(sample_indices=MagicMock())


def test_voice_mapper() -> None:
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
    model = VibeVoiceGeneration()

    # Custom voice parameter is returned in args
    result = await model.get_rest_args({"text": "Hello world", "voice": "custom_voice"})
    assert result["args"]["voice"] == "custom_voice"

    # Default voice returned when voice not specified
    result_default = await model.get_rest_args({"text": "Hello world"})
    assert "voice" in result_default["args"]
    assert result_default["args"]["voice"] == "af_heart"


@pytest.mark.asyncio
async def test_vibevoice_get_rest_args_voice_sample() -> None:
    """get_rest_args forwards voice_sample when present and omits it when absent."""
    model = VibeVoiceGeneration()

    with open("tests/data/sample.wav", "rb") as f:
        wav_bytes = f.read()
    dummy_audio = base64.b64encode(wav_bytes).decode()

    # voice_sample present -> included in args
    result = await model.get_rest_args({
        "text": "Hello world",
        "voice_sample": dummy_audio,
    })
    assert result["args"].get("voice_sample") == dummy_audio

    # voice_sample absent -> not included in args
    result_no_sample = await model.get_rest_args({"text": "Hello world"})
    assert "voice_sample" not in result_no_sample["args"]


def test_decode_voice_sample_to_tmp_file() -> None:
    """_decode_voice_sample_to_tmp_file writes the decoded bytes to a temp WAV file."""
    model = VibeVoiceGeneration()

    with open("tests/data/sample.wav", "rb") as f:
        audio_content = f.read()
    voice_sample_b64 = base64.b64encode(audio_content).decode()

    tmp_path = model._decode_voice_sample_to_tmp_file(voice_sample_b64)
    try:
        assert os.path.exists(tmp_path), "Temp file must be created"
        assert tmp_path.endswith(".wav"), "Temp file must have .wav extension"
        with open(tmp_path, "rb") as f:
            written = f.read()
        assert written == audio_content, "Written bytes must match original audio content"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_cleanup_tmp_voice_file() -> None:
    """_cleanup_tmp_voice_file removes the file and handles None / missing paths gracefully."""
    model = VibeVoiceGeneration()

    # None input is a no-op (must not raise)
    model._cleanup_tmp_voice_file(None)

    # Existing file is deleted
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    assert os.path.exists(tmp_path)
    model._cleanup_tmp_voice_file(tmp_path)
    assert not os.path.exists(tmp_path), "File must be deleted by _cleanup_tmp_voice_file"

    # Already-deleted path must not raise
    model._cleanup_tmp_voice_file(tmp_path)


def test_timestep_samplers() -> None:
    uniform = UniformSampler(timesteps=100)
    result_u = uniform.sample(batch_size=4, device=torch.device('cpu'))
    assert result_u is not None

    logit = LogitNormalSampler(timesteps=100)
    result_l = logit.sample(batch_size=4, device=torch.device('cpu'))
    assert result_l is not None


def test_dpm_solver_scheduler() -> None:
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
    for alpha_type in ["cosine", "exp", "cauchy", "laplace"]:
        betas = betas_for_alpha_bar(10, alpha_transform_type=alpha_type)
        assert betas is not None

    with pytest.raises(ValueError):
        betas_for_alpha_bar(10, alpha_transform_type="unknown")


def test_dpm_solver_set_timesteps() -> None:

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
    betas = torch.linspace(0.0001, 0.02, 100)
    rescaled = rescale_zero_terminal_snr(betas)
    assert rescaled is not None
    assert rescaled.shape == betas.shape

    # add_noise and get_velocity
    s.set_timesteps(10)
    original = torch.randn(2, 4)
    noise = torch.randn(2, 4)
    timesteps = torch.IntTensor([50, 80])
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
    model = VibeVoiceGeneration()

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
