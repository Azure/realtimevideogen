#!/usr/bin/env python3

import os
import sys
import pytest

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/4kagent")

# Only mock modules that are not available in the test environment
# (torch is replaced with a lightweight mock; nvidia_smi and colorlog
# need stubs since we have no GPU / colourised logging in CI).
# PyYAML and Pillow are real packages available in the test environment
# and must NOT be mocked so that file-I/O helpers work correctly.
mock_modules = {
    "torch": mock_torch,
    "nvidia_smi": MagicMock(),
    "colorlog": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from wrapper_4kagent import Upscale4KAgent


@pytest.mark.asyncio
async def test_wrapper_4kagent_init() -> None:
    """Test that Upscale4KAgent initialises with correct defaults."""
    model = Upscale4KAgent()
    assert model is not None
    assert model.model_name == "4kagent"
    assert model.status == "initializing"
    assert model.fourk_agent_dir is None
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_generate_before_init() -> None:
    """generate() must raise when the model has not been initialised."""
    model = Upscale4KAgent()
    img = Image.new("RGB", (64, 48))
    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(image=img)
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_init_ok() -> None:
    """load_model() succeeds when FOURK_AGENT_DIR exists."""
    model = Upscale4KAgent()

    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    assert model.status == "ok"
    assert model.fourk_agent_dir == Upscale4KAgent.FOURK_AGENT_DIR
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_init_missing_dir() -> None:
    """load_model() raises RuntimeError when FOURK_AGENT_DIR is absent."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=False):
        with pytest.raises(RuntimeError, match="4KAgent not found"):
            model.init()
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_get_rest_args_missing_json() -> None:
    """get_rest_args() raises on None or non-dict input."""
    model = Upscale4KAgent()
    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args("not a dict")  # type: ignore[arg-type]
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_get_rest_args_missing_img() -> None:
    """get_rest_args() raises when the 'img' key is absent."""
    model = Upscale4KAgent()
    with pytest.raises(ValueError, match="Missing 'img' parameter"):
        await model.get_rest_args({})
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_get_rest_args_success() -> None:
    """get_rest_args() returns the correct structure for a valid payload."""
    model = Upscale4KAgent()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({
        "job_id": "test-job",
        "img": img_base64,
        "profile_name": "ExpSR_s4_F",
        "tool_run_gpu_id": 1,
    })

    assert result["task"] == "4kagent"
    args = result["args"]
    assert args["job_id"] == "test-job"
    assert args["profile_name"] == "ExpSR_s4_F"
    assert args["tool_run_gpu_id"] == 1
    assert args["image"] is not None
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_get_rest_args_defaults() -> None:
    """get_rest_args() applies default profile_name and tool_run_gpu_id."""
    model = Upscale4KAgent()
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({"img": img_base64})

    assert result["args"]["profile_name"] == Upscale4KAgent.DEFAULT_PROFILE
    assert result["args"]["tool_run_gpu_id"] == 0
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_generate_no_image() -> None:
    """generate() raises ValueError when no image is provided."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    with pytest.raises(ValueError, match="input image is required"):
        await model.generate(image=None)
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_generate_success() -> None:
    """generate() calls the subprocess and returns the PIL result."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    expected_image = Image.new("RGB", (1024, 1024), color=(200, 200, 200))
    input_image = Image.new("RGB", (64, 48))

    with patch.object(model, "_run_4kagent_subprocess", new=AsyncMock()), \
         patch.object(model, "_load_result", return_value=expected_image):
        result = await model.generate(
            image=input_image,
            profile_name="ExpSR_s4_P",
            tool_run_gpu_id=0,
            job_id="unittest",
        )

    assert result is expected_image
    assert not model.running  # Must be cleared after completion
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_generate_subprocess_error() -> None:
    """generate() propagates subprocess failure as RuntimeError."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    async def fail_subprocess(**kwargs: object) -> None:
        raise RuntimeError("4KAgent subprocess failed (exit 1): error")

    input_image = Image.new("RGB", (64, 48))
    with patch.object(model, "_run_4kagent_subprocess", side_effect=fail_subprocess):
        with pytest.raises(RuntimeError, match="subprocess failed"):
            await model.generate(image=input_image)

    assert not model.running
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_health() -> None:
    """get_health() returns required keys including 4KAgent-specific ones."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    health = model.get_health()
    assert "model_name" in health
    assert "status" in health
    assert "fourk_agent_dir" in health
    assert "conda_env" in health
    assert health["conda_env"] == Upscale4KAgent.CONDA_ENV
    del model


@pytest.mark.asyncio
async def test_wrapper_4kagent_warmup() -> None:
    """warmup() delegates to generate() with a synthetic image."""
    model = Upscale4KAgent()
    with patch("os.path.isdir", return_value=True), \
         patch.object(model, "_write_config"):
        model.init()

    warmup_image = Image.new("RGB", (1024, 1024))
    with patch.object(model, "generate", new=AsyncMock(return_value=warmup_image)) as mock_gen:
        await model.warmup()
        mock_gen.assert_called_once()
        # The warmup call must pass an image argument
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("image") is not None or call_kwargs.args
    del model


def test_wrapper_4kagent_write_config() -> None:
    """_write_config() writes a YAML file populated from environment variables."""
    import yaml
    import tempfile

    model = Upscale4KAgent()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.fourk_agent_dir = tmpdir

        env = {
            "LLAMA_API_KEY": "test-llama-key",
            "OPENAI_API_KEY": "test-openai-key",
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_MODEL": "",
            "AZURE_OPENAI_API_VERSION": "",
        }
        with patch.dict(os.environ, env, clear=False):
            model._write_config()

        config_path = os.path.join(tmpdir, "config.yml")
        assert os.path.isfile(config_path)

        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        assert config["LLAMA"]["API_KEY"] == "test-llama-key"
        assert config["GPT"]["API_KEY"] == "test-openai-key"
        assert "AZUREGPT" in config

    del model


def test_load_result_no_results() -> None:
    """_load_result() raises ValueError when no result.png exists."""
    import tempfile

    model = Upscale4KAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No result.png found"):
            model._load_result(tmpdir)
    del model


def test_load_result_finds_image() -> None:
    """_load_result() returns a PIL Image from the expected output structure."""
    import tempfile

    model = Upscale4KAgent()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate 4KAgent output: <output_dir>/input/<step>/result.png
        result_dir = os.path.join(tmpdir, "input", "step_001")
        os.makedirs(result_dir)
        result_path = os.path.join(result_dir, "result.png")
        Image.new("RGB", (1024, 768), color=(10, 20, 30)).save(result_path)

        result = model._load_result(tmpdir)

    assert isinstance(result, Image.Image)
    assert result.size == (1024, 768)
    del model
