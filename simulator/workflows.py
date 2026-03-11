from __future__ import annotations

import math

from typing import Optional

from sim_types import WorkflowConfig
from sim_types import Model
from sim_types import QualityLevel

from constants import FPS
from constants import FRAMES_OPTIONS
from constants import FRAMES_PER_STEP_IDX
from constants import NUM_STEPS
from constants import SECONDS_IN_HOUR, SECONDS_IN_MINUTE
from constants import TOTAL_INPUT_TOKENS


# Shared physical constants
MAX_FT_FRAMES: int = 1 + 80
SUBSCENE_SECONDS: float = MAX_FT_FRAMES / FPS[Model.FT]  # 81 frames @ 23 FPS → ~3.52 s
SUBSCENES_PER_SCENE: int = 4  # default subscene grouping
TOKENS_PER_FRAME = 500  # 1 frame generates around 500 tokens


def _get_num_subscenes(total_video_seconds: int) -> int:
    """Return the number of subscenes needed to cover the given video duration."""
    return math.ceil(total_video_seconds / SUBSCENE_SECONDS)


def _get_num_scenes(total_video_seconds: int) -> int:
    """Return the number of scenes needed to cover the given video duration."""
    return math.ceil(_get_num_subscenes(total_video_seconds) / SUBSCENES_PER_SCENE)


def _get_num_frames(total_video_seconds: int, model: Model) -> int:
    """Return the number of frames needed for the given video duration and model."""
    return math.ceil(total_video_seconds * FPS[model])


def _video_gen_work(
    total_video_seconds: int,
    num_scenes: int,
    num_subscenes: int,
    model_work_overrides: Optional[dict[Model, int | str | None]] = None,
) -> dict[Model, int]:
    """Standard model work for video-generation workflows (Podcast, Movie, etc.)."""
    ret = {
        Model.GEMMA: 1,
        Model.FLUX: 1,
        Model.HF: num_subscenes,
        Model.HF_VAE: _get_num_frames(total_video_seconds, Model.HF),
        Model.FT: num_subscenes,
        Model.FT_VAE: _get_num_frames(total_video_seconds, Model.FT),
        Model.UPSCALER: _get_num_frames(total_video_seconds, Model.FT),
        Model.OTHERS: 1,
    }
    if model_work_overrides:
        for model, value in model_work_overrides.items():
            if value == "num_scenes":
                ret[model] = num_scenes
            elif value == "num_subscenes":
                ret[model] = num_subscenes
            elif isinstance(value, str):
                raise ValueError(f"Invalid model_work override value: {value}")
            elif value == 0 or value is None:
                del ret[model]
            else:
                ret[model] = value
    return ret


class WorkOverrideType:
    def __init__(self, value: int | str | None = None):
        self.value = value


def build_workflow_config(
    total_video_seconds: int,
    input_tokens: int,
    model_work: dict[Model, int] | None = None,
    *,
    model_work_overrides: dict[Model, int | str | None] | None = None,
    num_scenes_override: int | None = None,
    num_steps_override: dict[Model, int] | None = None,
    target_resolution: QualityLevel = QualityLevel.HIGH,
) -> WorkflowConfig:
    """Build a ``WorkflowConfig`` from base parameters, computing all derived values.

    Parameters
    ----------
    model_work:
        Explicit model-work dictionary.  When ``None`` (default), standard
        video-generation work is auto-generated from the other parameters.
    exclude_models:
        Models to remove from auto-generated ``model_work``.
    model_work_overrides:
        Key-value overrides applied on top of auto-generated ``model_work``.
        If a value is set to "num_scenes", it will be replaced with the number of scenes (i.e. per-scene work).
    target_resolution:
        The target output resolution for the workflow (default HIGH).
        When not HIGH, UPSCALER is automatically removed from model_work.
    """
    num_subscenes = _get_num_subscenes(total_video_seconds)

    num_scenes = _get_num_scenes(total_video_seconds)
    if num_scenes_override is not None:
        num_scenes = num_scenes_override

    num_steps = dict(NUM_STEPS)
    if num_steps_override:
        num_steps.update(num_steps_override)

    if model_work is None:
        model_work = _video_gen_work(
            total_video_seconds,
            num_scenes,
            num_subscenes,
            model_work_overrides,
        )

    return WorkflowConfig(
        total_video_seconds=total_video_seconds,
        total_scenes=num_scenes,
        total_subscenes=num_subscenes,
        total_frames={
            Model.HF: _get_num_frames(total_video_seconds, Model.HF),
            Model.FT: _get_num_frames(total_video_seconds, Model.FT),
        },
        per_subscene_frames={
            Model.HF: math.ceil(_get_num_frames(total_video_seconds, Model.HF) / num_subscenes),
            Model.FT: math.ceil(_get_num_frames(total_video_seconds, Model.FT) / num_subscenes),
        },
        num_steps=num_steps,
        hf_frames=FRAMES_OPTIONS[Model.HF],
        ft_frames=FRAMES_OPTIONS[Model.FT],
        frames_per_step_idx=FRAMES_PER_STEP_IDX,
        target_resolution=target_resolution,
        total_input_tokens=input_tokens,
        model_work=model_work,
    )


WORKFLOW_DURATIONS = {  # in seconds
    "podcast": int(10 * SECONDS_IN_MINUTE),
    # TODO The input is two hours but the output should be shorter something like 1 or 2 minutes
    "short": int(2 * SECONDS_IN_HOUR),
    "movie": int(2 * SECONDS_IN_HOUR),
    "story": int(10 * SECONDS_IN_MINUTE),
    "lecture": int(5 * SECONDS_IN_MINUTE),
    "slide": int(10 * SECONDS_IN_MINUTE),
    "dubbing": int(10 * SECONDS_IN_MINUTE),
    "editing": int(10 * SECONDS_IN_MINUTE),
    "chat": 5,
}


# Podcast: 10-minute video from text/PDF input
PODCAST_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["podcast"],
    input_tokens=TOTAL_INPUT_TOKENS,
)

# Shorts: short clips from a 2-hour input video
_SHORTS_SECONDS = WORKFLOW_DURATIONS["short"]
_SHORTS_SCENES = _SHORTS_SECONDS // 10  # 10-second scene segmentation → 720
SHORTS_WORKFLOW = build_workflow_config(
    total_video_seconds=_SHORTS_SECONDS,
    input_tokens=int(_SHORTS_SECONDS * TOKENS_PER_FRAME),  # 1 fps × 500 tokens/frame
    model_work={
        Model.GEMMA: _SHORTS_SCENES,
        Model.OTHERS: 1,  # TODO isn't this 1 by default?
    },
    num_scenes_override=_SHORTS_SCENES,
)

# Movie: 2-hour movie
MOVIE_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["movie"],
    input_tokens=TOTAL_INPUT_TOKENS,
    model_work_overrides={
        Model.FLUX: "num_scenes",
    },
)

# Animated Story: Podcast + 5% more HF denoising steps (LoRA overhead)
OVERHEAD_PCT = 5
ANIMATED_STORY_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["story"],
    input_tokens=TOTAL_INPUT_TOKENS,
    num_steps_override={
        Model.HF: int(NUM_STEPS[Model.HF] * 1 + (OVERHEAD_PCT / 100.0))
    },
)

# Lecture: 5-minute video, Flux generates per-scene images
LECTURE_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["lecture"],
    input_tokens=TOTAL_INPUT_TOKENS,
    model_work_overrides={
        Model.FLUX: "num_scenes",
    },
)

# Slide Persona: same as Podcast but at low resolution, no upscaler
SLIDE_PERSONA_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["slide"],
    input_tokens=TOTAL_INPUT_TOKENS,
    target_resolution=QualityLevel.LOW,
    model_work_overrides={
        Model.UPSCALER: None,
    },
)

# Dubbing: like Podcast but without Flux, and double the audio work
DUBBING_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["dubbing"],
    input_tokens=TOTAL_INPUT_TOKENS,
    model_work_overrides={
        Model.FLUX: None,
        Model.OTHERS: 2,  # Double audio work
    },
)

# Editing: like Podcast but without GEMMA, FLUX, or OTHERS
EDITING_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["editing"],
    input_tokens=TOTAL_INPUT_TOKENS,
    model_work_overrides={
        Model.GEMMA: None,
        Model.FLUX: None,
        Model.OTHERS: None,
    }
)

# Video Chat: like Podcast but only 5 seconds of output video
VIDEO_CHAT_WORKFLOW = build_workflow_config(
    total_video_seconds=WORKFLOW_DURATIONS["chat"],
    input_tokens=TOTAL_INPUT_TOKENS,
)


WORKFLOWS = {
    "podcast": PODCAST_WORKFLOW,
    "chat": VIDEO_CHAT_WORKFLOW,
    "dubbing": DUBBING_WORKFLOW,
    "editing": EDITING_WORKFLOW,
    "lecture": LECTURE_WORKFLOW,
    "movie": MOVIE_WORKFLOW,
    "short": SHORTS_WORKFLOW,
    "slide": SLIDE_PERSONA_WORKFLOW,
    "story": ANIMATED_STORY_WORKFLOW,
}
