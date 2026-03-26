"""
Wrapper for 4KAgent - agentic any-image-to-4K super-resolution.
https://github.com/taco-group/4KAgent
"""

import asyncio
import logging
import os
import sys
import tempfile

from pathlib import Path
from typing import override
from typing import Dict
from typing import Optional
from typing import Union
from typing import Any

import torch
import yaml

from PIL import Image

from wrapper_model import ModelGeneration

from image_utils import base64_to_img


class Upscale4KAgent(ModelGeneration):
    """
    Wrapper for 4KAgent image super-resolution.

    4KAgent is an agentic framework that upscales any image to 4K resolution.
    The 4KAgent code is imported directly from the cloned repository at
    FOURK_AGENT_DIR, which is added to sys.path during load_model().

    Required environment variables (at least one LLM key is needed):
        LLAMA_API_KEY   – Meta Llama API key (used by llama_vision profiles)
        OPENAI_API_KEY  – OpenAI API key (used by GPT-based profiles)
        AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_MODEL /
        AZURE_OPENAI_API_VERSION  – Azure OpenAI credentials
    """

    FOURK_AGENT_DIR: str = "/4kagent/4KAgent"
    DEFAULT_PROFILE: str = "ExpSR_s4_P"

    def __init__(self, model_name: str = "4kagent") -> None:
        super().__init__(model_name)
        self.fourk_agent_dir: Optional[str] = None

    def __del__(self) -> None:
        super().__del__()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        # 4KAgent manages its own GPU assignment internally.
        self.world_size = 1

        self.device_id = self.local_rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.device_id}")
        else:
            self.device = torch.device("cpu")

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        if not os.path.isdir(self.FOURK_AGENT_DIR):
            raise RuntimeError(
                f"4KAgent not found at {self.FOURK_AGENT_DIR}. "
                "Ensure the Docker image has cloned the repository."
            )
        self.fourk_agent_dir = self.FOURK_AGENT_DIR
        # Add the 4KAgent repo to sys.path so its packages can be imported
        # directly. This must happen before the first call to _run_4kagent().
        if self.fourk_agent_dir not in sys.path:
            sys.path.insert(0, self.fourk_agent_dir)
        self._write_config()

        # Warn if no LLM API key is configured.  Most llama_vision-based profiles
        # require LLAMA_API_KEY; GPT-based profiles need OPENAI_API_KEY or Azure
        # credentials.  A missing key will only cause a runtime failure when the
        # VLM is invoked, not during loading.
        llm_keys = [
            os.getenv("LLAMA_API_KEY", ""),
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("AZURE_OPENAI_API_KEY", ""),
        ]
        if not any(llm_keys):
            logging.warning(
                "No LLM API key found (LLAMA_API_KEY / OPENAI_API_KEY / "
                "AZURE_OPENAI_API_KEY).  Generation will fail unless the chosen "
                "profile does not require a VLM."
            )

        logging.info("Loaded 4KAgent from %s.", self.fourk_agent_dir)

    def _write_config(self) -> None:
        """Write config.yml populated with API keys from environment variables."""
        assert self.fourk_agent_dir is not None
        config: Dict[str, Any] = {
            "GPT": {
                "API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "MODEL": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
                "MAX_TOKENS": 3000,
                "TEMPERATURE": 0.0,
            },
            "LLAMA": {
                "API_KEY": os.getenv("LLAMA_API_KEY", ""),
                "MODEL": os.getenv("LLAMA_MODEL", "llama3.1-405b"),
                "MAX_TOKENS": 3000,
                "TEMPERATURE": 0.0,
            },
            "AZUREGPT": {
                "API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "MODEL": os.getenv("AZURE_OPENAI_MODEL", ""),
                "MAX_TOKENS": 3000,
                "TEMPERATURE": 0.0,
                "ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                "API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", ""),
            },
        }
        config_path = os.path.join(self.fourk_agent_dir, "config.yml")
        with open(config_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False)
        logging.info("Wrote 4KAgent config to %s.", config_path)

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("4KAgent does not support model parallelism.")

    def model_compile(self) -> None:
        # torch.compile is not applicable for direct-import execution.
        pass

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.fourk_agent_dir is None:
            raise ValueError("4KAgent directory not initialised.")

    @torch.inference_mode()
    async def warmup(self) -> None:
        logging.info("[%s] Warmup for 4KAgent.", self.rank)
        warmup_image = Image.new("RGB", (256, 192), color=(128, 128, 128))
        await self.generate(image=warmup_image)

    @override
    @torch.inference_mode()
    async def generate(
        self,
        image: Optional[Image.Image] = None,
        profile_name: str = DEFAULT_PROFILE,
        tool_run_gpu_id: int = 0,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Run 4KAgent super-resolution on *image* and return the upscaled result.

        Args:
            image: Input PIL image to upscale.
            profile_name: 4KAgent profile (e.g. "ExpSR_s4_P", "FastGen4K_P").
                          Profiles using ``llama_vision`` work without a separate
                          DepictQA server; profiles using ``depictqa`` require one.
            tool_run_gpu_id: GPU index for the restoration tool subprocesses.
            job_id: Optional job identifier for timing telemetry.

        Returns:
            Upscaled PIL image.
        """
        gen_timer = self._new_gen_timer(job_id)
        self._assert_model_init()

        if image is None:
            raise ValueError("An input image is required for 4KAgent generation.")

        self.running = True
        try:
            with tempfile.TemporaryDirectory(prefix="4kagent_in_") as input_dir:
                with tempfile.TemporaryDirectory(prefix="4kagent_out_") as output_dir:
                    gen_timer.start("save_input")
                    input_path = Path(input_dir) / "input.png"
                    await asyncio.to_thread(image.save, str(input_path))
                    gen_timer.end("save_input")

                    gen_timer.start("inference")
                    await asyncio.to_thread(
                        self._run_4kagent,
                        input_path,
                        Path(output_dir),
                        profile_name,
                        tool_run_gpu_id,
                    )
                    gen_timer.end("inference")

                    # Load and return the result before the temp dirs are removed.
                    gen_timer.start("load_result")
                    result = await asyncio.to_thread(self._load_result, output_dir)
                    gen_timer.end("load_result")

                    return result
        finally:
            self.running = False
            gen_timer.end("total")

    def _run_4kagent(
        self,
        input_path: Path,
        output_dir: Path,
        profile_name: str,
        tool_run_gpu_id: int,
    ) -> None:
        """Import and run The4KAgent pipeline directly.

        Called via asyncio.to_thread; 4KAgent handles one image at a time so
        concurrent requests are serialised by the caller's semaphore.
        The lazy import (after sys.path is set in load_model) is intentional:
        Python's import system caches modules in sys.modules so subsequent
        calls incur only a dict lookup.
        """
        assert self.fourk_agent_dir is not None
        from pipeline.the4kagent_pipeline import The4KAgent  # noqa: PLC0415
        logging.info(
            "[%s] Running 4KAgent: profile=%s, tool_gpu=%s.",
            self.rank, profile_name, tool_run_gpu_id,
        )
        agent = The4KAgent(
            input_path=input_path,
            output_dir=output_dir,
            llm_config_path=Path(self.fourk_agent_dir) / "config.yml",
            with_retrieval=True,
            with_reflection=True,
            silent=False,
            tool_run_gpu_id=tool_run_gpu_id,
            profile_name=profile_name,
        )
        agent.run()

    def _load_result(self, output_dir: str) -> Image.Image:
        """
        Locate and load the result image produced by 4KAgent.

        4KAgent writes to ``<output_dir>/<image_stem>/<step>/result.png``.
        """
        result_candidates = sorted(Path(output_dir).glob("*/*/result.png"))
        if not result_candidates:
            raise ValueError(
                f"No result.png found under {output_dir}. "
                "4KAgent may have failed or the output structure changed."
            )
        result_path = result_candidates[-1]
        logging.info("[%s] Loading result from %s.", self.rank, result_path)
        # .copy() ensures the image is fully loaded before the temp dir is removed.
        return Image.open(result_path).copy()

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "rank": self.rank,
            "world_size": self.world_size,
            "fourk_agent_dir": self.fourk_agent_dir,
        })
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]],
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter")

        image = base64_to_img(str(img_base64))

        return {
            "task": self.model_name,
            "args": {
                "job_id": data_json.get("job_id", None),
                "image": image,
                "profile_name": str(data_json.get("profile_name", self.DEFAULT_PROFILE)),
                "tool_run_gpu_id": int(data_json.get("tool_run_gpu_id", 0)),
            },
        }
