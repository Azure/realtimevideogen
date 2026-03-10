"""
Base class for model generation wrappers.
"""
import torch
import logging
import time
import traceback

import nvidia_smi
from nvidia_smi import NVMLError

from typing import List
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from abc import ABC
from abc import abstractmethod

from datetime import datetime

from console_utils import setup_logging

from model_timing import LoadTimer
from model_timing import GenTimer


class GenerationInterruptedError(Exception):
    """Exception raised when generation is interrupted."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelGeneration(ABC):
    """Base class for model generation."""

    def __init__(
        self,
        model_name: str,
        torch_compile: bool = True,
    ) -> None:
        self.running = False
        self.interrupted = False
        self.status = "initializing"
        self.model_name = model_name

        # Parallelism
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device_id: Union[int, str] = 0
        self.device: Optional[torch.device] = None

        self.torch_compile = torch_compile

        # Timing
        self.load_timer = LoadTimer()
        self.gen_timers: Dict[str, GenTimer] = {}  # id -> GenTimer

        self.gpu_setup = True

    def __del__(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def interrupt(self) -> None:
        """Interrupt the current generation process."""
        if self.world_size > 1:
            logging.warning(f"[{self.rank}] Interruption not supported in multi-GPU mode.")

        if self.running:
            logging.info(f"[{self.rank}] Interrupting.")
            self.interrupted = True
        else:
            logging.warning(f"[{self.rank}] Not running, cannot interrupt.")

    def init_logging(self) -> None:
        """Initialize logging with colored output."""
        setup_logging(
            path="/tmp",
            file_name="streamwise.log",
            level=logging.INFO,
        )

    def init(self) -> None:
        """Initialize the model, including loading and setting up parallelism."""
        t0 = time.time()

        try:
            self.status = "initializing parallelism"
            self.init_parallelism()

            self.init_logging()

            self.status = "loading model"
            self.load_model()

            self.status = "parallelizing model"
            self.init_model_parallelism()

            self.status = "compiling model"
            self.model_compile()

            self.status = "ok"

            logging.info(f"[{self.rank}] Model loaded in {time.time() - t0:.3f} seconds.")

            torch.cuda.empty_cache()  # for tracking memory properly
            mem_gb = torch.cuda.memory_allocated() / 1024 / 1024 ** 2
            logging.info(f"[{self.rank}] Total memory allocated: {mem_gb:.2f} GB.")
        except Exception as ex:
            logging.error(f"Error during initialization: {ex}.")
            logging.error(f"Trace: {traceback.format_exc()}.")
            self.status = "failed"
            raise ex
        finally:
            self.load_timer.end()

    def init_parallelism(self) -> None:
        """Initialize distributed parallelism if applicable."""
        logging.info("Parallelism initialization should be implemented in subclass.")

    def load_model(self) -> None:
        """Load the model into memory."""
        logging.info("Model initialization should be implemented in subclass.")

    def init_model_parallelism(self) -> None:
        """Set up model parallelism if applicable."""
        logging.info("Model parallelism should be implemented in subclass.")

    def model_compile(self) -> None:
        """Compile the model using torch.compile if enabled."""
        logging.info("Model parallelism should be implemented in subclass.")

    def _assert_model_init(self) -> None:
        """Assert that the model is initialized and ready."""
        if self.status != "ok":
            raise ValueError(f"Model not initialized. Current status: {self.status}.")

    def get_gpu_info(self) -> Optional[List[Dict[str, Any]]]:
        """Get information about the GPUs on the system."""
        if not self.gpu_setup:
            return None

        ret = []
        try:
            nvidia_smi.nvmlInit()
            device_count = nvidia_smi.nvmlDeviceGetCount()
            if device_count == 0:
                logging.warning("No GPUs found.")
                self.gpu_setup = False
                return None

            local_gpu_index = torch.cuda.current_device()

            for gpu_index in range(device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name_raw = nvidia_smi.nvmlDeviceGetName(handle)
                gpu_name: str
                if isinstance(gpu_name_raw, bytes):
                    gpu_name = gpu_name_raw.decode("utf-8")
                else:
                    gpu_name = gpu_name_raw
                gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

                ret.append({
                    "index": gpu_index,
                    "current": gpu_index == local_gpu_index,
                    "name": gpu_name,
                    "sm_util": gpu_util.gpu,
                    "mem_util": gpu_util.memory,
                    "mem_gib_used": gpu_mem_info.used / (1024 ** 3),
                    "mem_gib_total": gpu_mem_info.total / (1024 ** 3),
                    "temp": nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU),
                    # Power in Watts
                    "power_draw_watts": nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    "power_limit_watts": nvidia_smi.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0,
                    # Frequencies in MHz
                    "graphics_clock": nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_GRAPHICS),
                    "sm_clock": nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_SM),
                    "mem_clock": nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_MEM),
                })

            return ret
        except Exception as ex:
            logging.warning(f"Error getting GPU info: {ex}.")
            self.gpu_setup = False  # Disable GPU info retrieval on error
            return None
        finally:
            try:
                nvidia_smi.nvmlShutdown()
            except NVMLError as nvml_err:
                logging.warning(f"Error shutting down NVML: {nvml_err}.")

    def _new_gen_timer(
        self,
        job_id: Optional[str] = None
    ) -> GenTimer:
        if job_id is None:
            job_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
        elif job_id in self.gen_timers:
            new_job_id = job_id + "_" + datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            logging.info(f"[{self.rank}] Job '{job_id}' already exists, using '{new_job_id}'.")
            job_id = new_job_id
        gen_timer = GenTimer()
        self.gen_timers[job_id] = gen_timer
        return gen_timer

    def get_health(self) -> Dict[str, Any]:
        """Get health status of the model."""
        ret = {
            "model_name": self.model_name,
            "status": self.status,
            "running": self.running,
            "load_timer": self.load_timer.to_dict() if self.load_timer else None,
            "gen_timer": {
                job_id: gen_timer.to_dict()
                for job_id, gen_timer in self.gen_timers.items()
            } if self.gen_timers else None,
        }

        gpu_info = self.get_gpu_info()
        if gpu_info:
            ret["gpu_info"] = gpu_info

        return ret

    def get_timestamps(self) -> List[dict]:
        """Get timing timestamps for loading and generation."""
        ret = []
        if self.load_timer:
            timestamps = self.load_timer.to_timestamps(
                group=self.model_name,
                subgroup="load")
            if timestamps:
                ret.extend(timestamps)
        if self.gen_timers:
            for gen_ix, (job_id, gen_timer) in enumerate(self.gen_timers.items()):
                timestamps = gen_timer.to_timestamps(
                    group=self.model_name,
                    subgroup=job_id or f"gen_{gen_ix:04d}")
                if timestamps:
                    ret.extend(timestamps)
        return ret

    @abstractmethod
    async def generate(
        self,
        job_id: Optional[str] = None,
        *args: Any,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """Generate output using the model."""
        raise NotImplementedError("Method should be implemented in subclasses.")

    @abstractmethod
    async def warmup(self) -> None:
        """Warmup the model with a sample generation."""
        raise NotImplementedError("Method should be implemented in subclasses.")

    @abstractmethod
    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        """Extract and validate arguments from the REST API request."""
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        raise NotImplementedError("Method should be implemented in subclasses.")
