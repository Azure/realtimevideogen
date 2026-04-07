import os
import sys
import datetime
import random
import logging

import torch
import torch.distributed as dist

from typing import Dict
from typing import Any
from typing import Optional
from typing import Tuple

from wrapper_model import ModelGeneration

if not torch.cuda.is_available():
    raise RuntimeError("This module requires CUDA support.")

from xfuser.config import EngineConfig
from xfuser.core.distributed import initialize_model_parallel
from xfuser.core.distributed import init_distributed_environment


class USPGeneration(ModelGeneration):
    """
    Base class for generation using USP.
    This models support Unified Sequence Parallelism (USP).
    """

    def __init__(
        self,
        model_name: str,
        engine_config: Optional[EngineConfig] = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        # Model components
        self.engine_config = engine_config
        self.ulysses_size = -1
        self.ring_size = -1
        if self.engine_config and getattr(self.engine_config, "parallel_config", None):
            self.ulysses_size = self.engine_config.parallel_config.sp_config.ulysses_degree
            self.ring_size = self.engine_config.parallel_config.sp_config.ring_degree

        if self.engine_config and getattr(self.engine_config, "runtime_config", None):
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile

        self.param_dtype = param_dtype

        # Parallelism
        self.gpu: Optional[str] = None
        if torch.cuda.is_available():
            self.gpu = torch.cuda.get_device_name(0)

        self.base_seed = random.randint(0, sys.maxsize)

        # Model features
        self.num_heads = -1
        self.vae_stride: Optional[Tuple[int, int, int]] = None  # time, height, width

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().__del__()

    def set_seed(
        self,
        seed: int
    ) -> None:
        """Set the seed for random number generation."""
        self.base_seed = seed

        if dist.is_initialized():
            global_base_seed: list[int | None] = [self.base_seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(global_base_seed, src=0)
            seed_value = global_base_seed[0]
            assert seed_value is not None
            self.base_seed = seed_value
        logging.info(f"[{self.rank}] Using base seed: {self.base_seed}")

    def reset_seed(self) -> None:
        """Reset the seed for random number generation."""
        rnd_seed = random.randint(0, sys.maxsize)
        self.set_seed(rnd_seed)

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank

        if not torch.cuda.is_available():
            self.device_id = 0
            self.device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU.")
            self.load_timer.end("torch_dist")
            return  # Single GPU mode, no parallelism needed

        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size <= 1:
            self.load_timer.end("torch_dist")
            return  # Single GPU mode, no parallelism needed

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
                timeout=datetime.timedelta(hours=24),  # Prevent NCCL timeout
            )

        # Unified Sequence Parallelism (USP)
        if self.ulysses_size * self.ring_size != self.world_size:
            raise ValueError(
                f"ulysses_size {self.ulysses_size} x ring_size {self.ring_size} != world size {self.world_size}.")

        if self.ulysses_size > 1 and self.num_heads > 0:
            if self.num_heads % self.ulysses_size != 0:
                raise ValueError(f"`{self.num_heads}` cannot be divided evenly by `{self.ulysses_size}`.")

        if dist.is_initialized():
            self.reset_seed()

        self.load_timer.end("torch_dist")

        self.load_timer.start("init_distributed")
        init_distributed_environment(
            rank=dist.get_rank(),
            world_size=dist.get_world_size()
        )
        self.load_timer.end("init_distributed")

        self.load_timer.start("model_parallel")
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ulysses_degree=self.ulysses_size,
            ring_degree=self.ring_size,
        )
        self.load_timer.end("model_parallel")

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group not initialized")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.gpu,
            "rank": self.rank,
            "world_size": self.world_size,
            "ulysses_size": self.ulysses_size,
            "ring_size": self.ring_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            "vae_stride": self.vae_stride,
            "num_heads": self.num_heads,
        })
        return ret
