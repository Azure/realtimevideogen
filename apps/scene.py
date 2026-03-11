"""
Scene information for video generation apps.
TODO consolidate info and segment.
TODO make it JSON encode/decode friendly for saving/loading from disk or database
"""

from typing import List
from typing import Optional

from dataclasses import dataclass
from dataclasses import field


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class SceneSegment:
    """A scene segment in a video."""
    scene_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float

    frame_image_paths: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)

    audio_path: Optional[str] = None
    transcript: Optional[str] = None

    def add_image_path(
        self,
        image_path: str
    ) -> None:
        if image_path:
            self.frame_image_paths.append(image_path)

    def add_description(
        self,
        description: str
    ) -> None:
        if description:
            self.descriptions.append(description)

    def get_start(self) -> str:
        return format_time(self.start_sec)

    def get_end(self) -> str:
        return format_time(self.end_sec)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def __str__(self) -> str:
        ret = f"[{self.start_frame:4d}-{self.end_frame:4d}, {self.start_sec:4.1f}-{self.end_sec:4.1f}]"
        if self.transcript:
            ret += f": {self.transcript[0:60]}..."
        for description in self.descriptions:
            ret += f" | {description[0:60]}..."
        if self.frame_image_paths:
            ret += f" | {len(self.frame_image_paths)} images"
        return ret
