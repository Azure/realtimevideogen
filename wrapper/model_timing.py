import time
import math
import torch

from typing import List
from typing import Dict
from typing import Optional
from typing import Any


class TimePeriod:
    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all prior CUDA ops are done
        self.end_time = time.time()

    def get_seconds(self) -> float:
        if self.end_time is None:
            return -1.0
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return f"{self.get_seconds():.2f}"

    # TypeError: Object of type TimePeriod is not JSON serializable
    def to_dict(self) -> Dict[str, float]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time or -1.0,
            "duration_seconds": round(self.get_seconds(), 2)
        }


class Timer:
    def __init__(self) -> None:
        self.timing = {}
        self.timing["total"] = TimePeriod()

    def start(self, event_name: str = "total") -> None:
        self.timing[event_name] = TimePeriod()

    def end(self, event_name: str = "total") -> None:
        if event_name not in self.timing:
            raise ValueError(f"Event {event_name} not found in {self.timing.keys()}.")
        self.timing[event_name].end()

    def get_last_event_name(self) -> Optional[str]:
        if not self.timing:
            return None
        # Assumes that keys are added in order
        return list(self.timing.keys())[-1]

    def get_total_seconds(self) -> float:
        if "total" not in self.timing:
            return -1.0
        return self.timing["total"].get_seconds()

    def __str__(self) -> str:
        str_ret = ""
        for key, val in self.timing.items():
            str_ret += f"{key}: {val.get_seconds():.3f}, "
        return str_ret[:-2]

    def to_dict(self) -> dict:
        return {k: round(v.get_seconds(), 2) for k, v in self.timing.items()}

    def to_timestamps(
        self,
        group: Optional[str] = None,
        subgroup: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        events = []
        for key, val in self.timing.items():
            id_key = f"{group}_{key}" if group else key
            if subgroup:
                id_key = f"{subgroup}_{key}"
            event = {
                "id": id_key,
                "content": key,
                # ceil and floor to nearest ms to avoid overlap, keep in seconds
                "start": math.ceil(val.start_time * 1000) / 1000 if val.start_time is not None else None,
                "end": math.floor(val.end_time * 1000) / 1000 if val.end_time is not None else None,
                "duration_seconds": val.get_seconds()
            }
            if group:
                event["group"] = group
            if subgroup:
                event["subgroup"] = subgroup
                event["className"] = subgroup
            events.append(event)
        return events


class LoadTimer(Timer):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        '''
        For video generation:
            text_encoder
            image_encoder
            vae
            dit
        '''
        if "text_encoder" not in self.timing:
            return ""

        return f"{self.timing['text_encoder'].get_seconds():.3f}," + \
            "{self.timing['image_encoder'].get_seconds():.3f}," + \
            "{self.timing['vae'].get_seconds():.3f}," + \
            "{self.timing['dit'].get_seconds():.3f}," + \
            "{self.get_total_seconds():.3f}"


class GenTimer(Timer):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        '''
        For video/image generation, the order is:
            text_encoder
            image_encoder
            vae_encoder
            scheduler_setup
            dit_{it}
            dit_{it}_{it}
            scheduler_{it}
            vae_decoder
            video_generation
        '''
        if "text_encoder" not in self.timing:
            return ""

        str_ret = f"{self.timing['text_encoder'].get_seconds():.3f}," + \
            "{self.timing['image_encoder'].get_seconds():.3f}," + \
            "{self.timing['vae_encoder'].get_seconds():.3f}," + \
            "{self.timing['scheduler_setup'].get_seconds():.3f},"
        dit_time = 0.0
        scheduler_time = 0.0
        for key, val in self.timing.items():
            if key.startswith("dit_"):
                dit_time += val.get_seconds()
            elif key.startswith("scheduler_"):
                scheduler_time += val.get_seconds()
        str_ret += f"{dit_time:.3f},{scheduler_time:.3f},"
        str_ret += f"{self.timing['vae_decoder'].get_seconds():.3f},{self.get_total_seconds():.3f}"
        return str_ret
