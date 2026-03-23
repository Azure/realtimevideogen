"""
Code not available yet
https://github.com/taco-group/4KAgent
"""

import logging
import os

import torch
import torch.distributed as dist

from typing import override
from typing import Dict
from typing import Optional
from typing import Any

from wrapper_model import ModelGeneration

from xfuser.config import EngineConfig


class Upscale4KAgent(ModelGeneration):
    def __init__(
        self,
        model_name: str = "4kagent",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        # TODO
        self.model: Optional[Any] = None

    def __del__(self) -> None:
        # TODO
        if self.model is not None:
            del self.model
            self.model = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            logging.warning("LlamaGen is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        # TODO

        logging.info("Loaded 4KAgent.")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("4KAgent does not support model parallelism yet.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        self.load_timer.start("model_compile")
        # TODO
        self.load_timer.end("model_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.model is not None

    def _assert_args(self) -> None:
        pass

    @torch.inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for LlamaGen generation.")
        await self.generate()

    @override
    @torch.inference_mode()
    async def generate(
        self,
        job_id: Optional[str] = None
    ) -> None:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args()

        self.running = True  # Mark running to avoid concurrent calls

        try:
            return None
        finally:
            self.running = False
            gen_timer.end("total")

    def get_health(self) -> Dict:
        ret = super().get_health()
        ret.update({
            "rank": self.rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            # TODO
        })
        return ret

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")
        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")

        # TODO
        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                # TODO
            }
        }
