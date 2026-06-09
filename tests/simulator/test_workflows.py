"""
Unit tests for simulator/workflows.py

Tests for build_workflow_config, _video_gen_work, and the pre-built
workflow configs (PODCAST_WORKFLOW, SHORTS_WORKFLOW, etc.).
"""

import math
import sys
import os

import pytest

sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import WorkflowConfig, Model, QualityLevel, GPUType
    from constants import (
        FPS,
        FRAMES_OPTIONS,
        FRAMES_PER_STEP_IDX,
        NUM_STEPS,
        SECONDS_IN_HOUR,
        SECONDS_IN_MINUTE,
        TOTAL_INPUT_TOKENS,
    )
    from data_loading import load_latency_data
    from auto_model_allocator import AutoModelAllocator
    from model_provisioner.policies import STREAMWISE_POLICY, NAIVE_POLICY
    from workflows import (
        MAX_FT_FRAMES,
        SUBSCENE_SECONDS,
        SUBSCENES_PER_SCENE,
        build_workflow_config,
        _get_num_subscenes,
        _get_num_scenes,
        _video_gen_work,
        WORKFLOWS,
        PODCAST_WORKFLOW,
        SHORTS_WORKFLOW,
        MOVIE_WORKFLOW,
        ANIMATED_STORY_WORKFLOW,
        LECTURE_WORKFLOW,
        DUBBING_WORKFLOW,
        EDITING_WORKFLOW,
        VIDEO_CHAT_WORKFLOW,
    )


class TestModuleConstants:
    """Module-level constant tests."""
    def test_max_ft_frames(self) -> None:
        assert MAX_FT_FRAMES == 1 + 80

    def test_subscene_seconds(self) -> None:
        assert SUBSCENE_SECONDS == pytest.approx(MAX_FT_FRAMES / FPS[Model.FT])

    def test_subscenes_per_scene(self) -> None:
        assert SUBSCENES_PER_SCENE == 4


class TestHelperFunctions:
    """Helper function tests."""
    def test_num_subscenes(self) -> None:
        secs = 2 * 60  # 2 minutes
        assert _get_num_subscenes(secs) == math.ceil(secs / SUBSCENE_SECONDS)

    def test_num_scenes(self) -> None:
        secs = 2 * 60  # 2 minutes
        expected_subscenes = math.ceil(secs / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert _get_num_scenes(secs) == expected_scenes


class TestBuildWorkflowConfig:
    """Build workflow config tests."""

    def test_returns_workflow_config(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=1000,
            model_work={},
        )
        assert isinstance(cfg, WorkflowConfig)

    def test_basic_fields(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=1000,
            model_work={},
        )
        num_ss = _get_num_subscenes(60)
        assert cfg.total_video_seconds == 60
        assert cfg.total_scenes == _get_num_scenes(60)
        assert cfg.total_subscenes == num_ss
        assert cfg.total_input_tokens == 1000
        assert cfg.target_resolution == QualityLevel.HIGH
        assert cfg.hf_frames == FRAMES_OPTIONS[Model.HF]
        assert cfg.ft_frames == FRAMES_OPTIONS[Model.FT]
        assert cfg.frames_per_step_idx == FRAMES_PER_STEP_IDX

    def test_total_frames(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=0,
            model_work={},
        )
        assert cfg.total_frames[Model.HF] == 60 * FPS[Model.HF]
        assert cfg.total_frames[Model.FT] == 60 * FPS[Model.FT]

    def test_per_subscene_frames(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=0,
            model_work={},
        )
        video_seconds = 60
        num_ss = _get_num_subscenes(video_seconds)
        assert cfg.per_subscene_frames[Model.HF] == math.ceil(video_seconds * FPS[Model.HF] / num_ss)
        assert cfg.per_subscene_frames[Model.FT] == math.ceil(video_seconds * FPS[Model.FT] / num_ss)

    def test_num_steps_default(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=0,
            model_work={},
        )
        assert cfg.num_steps[Model.FLUX] == NUM_STEPS[Model.FLUX]
        assert cfg.num_steps[Model.HF] == NUM_STEPS[Model.HF]
        assert cfg.num_steps[Model.FT] == NUM_STEPS[Model.FT]

    def test_num_scenes_override(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=0,
            model_work={},
            num_scenes_override=42,
        )
        assert cfg.total_scenes == 42

    def test_num_steps_override(self) -> None:
        cfg = build_workflow_config(
            total_video_seconds=60,
            input_tokens=0,
            model_work={},
            num_steps_override={Model.HF: 99},
        )
        assert cfg.num_steps[Model.HF] == 99
        # Other steps unchanged
        assert cfg.num_steps[Model.FLUX] == NUM_STEPS[Model.FLUX]
        assert cfg.num_steps[Model.FT] == NUM_STEPS[Model.FT]


class TestPodcastWorkflow:
    """Podcast workflow config tests."""

    PODCAST_TOTAL_SECONDS = int(10 * SECONDS_IN_MINUTE)  # 600 s

    def test_total_video_seconds(self) -> None:
        assert PODCAST_WORKFLOW.total_video_seconds == self.PODCAST_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert PODCAST_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.PODCAST_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert PODCAST_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.PODCAST_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert PODCAST_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(PODCAST_WORKFLOW.model_work.keys()) == expected_models

    def test_model_work_singleton_values(self) -> None:
        assert PODCAST_WORKFLOW.model_work[Model.GEMMA] == 1
        assert PODCAST_WORKFLOW.model_work[Model.FLUX] == 1
        assert PODCAST_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_model_work_subscene_values(self) -> None:
        assert PODCAST_WORKFLOW.model_work[Model.HF] == PODCAST_WORKFLOW.total_subscenes
        assert PODCAST_WORKFLOW.model_work[Model.FT] == PODCAST_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert PODCAST_WORKFLOW.model_work[Model.HF_VAE] == self.PODCAST_TOTAL_SECONDS * FPS[Model.HF]
        assert PODCAST_WORKFLOW.model_work[Model.UPSCALER] == self.PODCAST_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(PODCAST_WORKFLOW.models) == {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_parallelizable_models(self) -> None:
        # HF and FT have work == num_subscenes (>1), so parallelizable
        assert PODCAST_WORKFLOW.is_parallelizable(Model.HF)
        assert PODCAST_WORKFLOW.is_parallelizable(Model.FT)
        # VAE and UPSCALER have large work counts
        assert PODCAST_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert PODCAST_WORKFLOW.is_parallelizable(Model.UPSCALER)
        # Singleton models are not parallelizable
        assert not PODCAST_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not PODCAST_WORKFLOW.is_parallelizable(Model.FLUX)
        assert not PODCAST_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.PODCAST_TOTAL_SECONDS
        num_scenes = _get_num_scenes(secs)
        num_subscenes = _get_num_subscenes(secs)
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=_video_gen_work(
                secs,
                num_scenes,
                num_subscenes,
                model_work_overrides={Model.FLUX: 1},
            ),
        )
        assert PODCAST_WORKFLOW == fresh


class TestShortsWorkflow:
    """Shorts workflow config tests."""
    SHORTS_TOTAL_SECONDS = int(2 * SECONDS_IN_HOUR)  # 7200 s

    def test_total_video_seconds(self) -> None:
        assert SHORTS_WORKFLOW.total_video_seconds == self.SHORTS_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        expected = int(2 * SECONDS_IN_HOUR * 500)
        assert SHORTS_WORKFLOW.total_input_tokens == expected

    def test_num_scenes_override(self) -> None:
        """ShortsWorkflow overrides num_scenes to derive from input video."""
        assert SHORTS_WORKFLOW.total_scenes == self.SHORTS_TOTAL_SECONDS // 10
        assert SHORTS_WORKFLOW.total_scenes == 720

    def test_num_subscenes_uses_base(self) -> None:
        """num_subscenes still uses the base formula."""
        assert SHORTS_WORKFLOW.total_subscenes == math.ceil(self.SHORTS_TOTAL_SECONDS / SUBSCENE_SECONDS)

    def test_model_work_keys(self) -> None:
        assert set(SHORTS_WORKFLOW.model_work.keys()) == {Model.GEMMA, Model.OTHERS}

    def test_model_work_values(self) -> None:
        assert SHORTS_WORKFLOW.model_work[Model.GEMMA] == SHORTS_WORKFLOW.total_scenes
        assert SHORTS_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_gemma_parallelizable(self) -> None:
        # 720 scenes → GEMMA work = 720, parallelizable
        assert SHORTS_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not SHORTS_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_config_models_only_gemma_and_others(self) -> None:
        assert set(SHORTS_WORKFLOW.models) == {Model.GEMMA, Model.OTHERS}

    def test_no_video_generation_models(self) -> None:
        """Shorts workflow does not include video generation models."""
        for model in (Model.FLUX, Model.HF, Model.HF_VAE, Model.FT, Model.FT_VAE, Model.UPSCALER):
            assert model not in SHORTS_WORKFLOW.model_work

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.SHORTS_TOTAL_SECONDS
        scenes = secs // 10
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=int(2 * SECONDS_IN_HOUR * 500),
            model_work={Model.GEMMA: scenes, Model.OTHERS: 1},
            num_scenes_override=scenes,
        )
        assert SHORTS_WORKFLOW == fresh


class TestMovieWorkflow:
    """Movie workflow config tests."""
    MOVIE_TOTAL_SECONDS = int(2 * SECONDS_IN_HOUR)  # 7200 s

    def test_total_video_seconds(self) -> None:
        assert MOVIE_WORKFLOW.total_video_seconds == self.MOVIE_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert MOVIE_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.MOVIE_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert MOVIE_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.MOVIE_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert MOVIE_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(MOVIE_WORKFLOW.model_work.keys()) == expected_models

    def test_model_work_singleton_values(self) -> None:
        assert MOVIE_WORKFLOW.model_work[Model.GEMMA] == 1
        assert MOVIE_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_model_work_flux_per_scene(self) -> None:
        """Movie generates one FLUX image per scene (unlike Podcast which uses 1)."""
        assert MOVIE_WORKFLOW.model_work[Model.FLUX] == MOVIE_WORKFLOW.total_scenes

    def test_model_work_subscene_values(self) -> None:
        assert MOVIE_WORKFLOW.model_work[Model.HF] == MOVIE_WORKFLOW.total_subscenes
        assert MOVIE_WORKFLOW.model_work[Model.FT] == MOVIE_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert MOVIE_WORKFLOW.model_work[Model.HF_VAE] == self.MOVIE_TOTAL_SECONDS * FPS[Model.HF]
        assert MOVIE_WORKFLOW.model_work[Model.UPSCALER] == self.MOVIE_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(MOVIE_WORKFLOW.models) == {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_flux_parallelizable(self) -> None:
        """Movie has FLUX work == num_scenes (>1), so it's parallelizable."""
        assert MOVIE_WORKFLOW.is_parallelizable(Model.FLUX)

    def test_parallelizable_models(self) -> None:
        assert MOVIE_WORKFLOW.is_parallelizable(Model.HF)
        assert MOVIE_WORKFLOW.is_parallelizable(Model.FT)
        assert MOVIE_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert MOVIE_WORKFLOW.is_parallelizable(Model.UPSCALER)
        assert MOVIE_WORKFLOW.is_parallelizable(Model.FLUX)
        assert not MOVIE_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not MOVIE_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.MOVIE_TOTAL_SECONDS
        num_sc = _get_num_scenes(secs)
        num_ss = _get_num_subscenes(secs)
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=_video_gen_work(
                secs,
                num_sc,
                num_ss,
                model_work_overrides={Model.FLUX: "num_scenes"},
            ),
        )
        assert MOVIE_WORKFLOW == fresh


class TestWorkflowComparisons:
    """Cross-workflow comparison tests."""

    def test_movie_more_subscenes_than_podcast(self) -> None:
        assert MOVIE_WORKFLOW.total_subscenes > PODCAST_WORKFLOW.total_subscenes

    def test_movie_more_scenes_than_podcast(self) -> None:
        assert MOVIE_WORKFLOW.total_scenes > PODCAST_WORKFLOW.total_scenes

    def test_shorts_has_most_scenes(self) -> None:
        """Shorts derives scenes from 2h input → 720 scenes, more than movie."""
        assert SHORTS_WORKFLOW.total_scenes > MOVIE_WORKFLOW.total_scenes

    def test_shorts_fewer_model_types_than_podcast(self) -> None:
        assert len(SHORTS_WORKFLOW.model_work) < len(PODCAST_WORKFLOW.model_work)

    def test_movie_and_podcast_same_model_types(self) -> None:
        assert set(MOVIE_WORKFLOW.model_work.keys()) == set(PODCAST_WORKFLOW.model_work.keys())

    def test_movie_flux_work_exceeds_podcast(self) -> None:
        """Movie needs FLUX per scene; Podcast only needs 1."""
        assert MOVIE_WORKFLOW.model_work[Model.FLUX] > PODCAST_WORKFLOW.model_work[Model.FLUX]

    def test_movie_vae_work_exceeds_podcast(self) -> None:
        assert MOVIE_WORKFLOW.model_work[Model.HF_VAE] > PODCAST_WORKFLOW.model_work[Model.HF_VAE]


class TestAnimatedStoryWorkflow:
    """Animated Story workflow config tests."""
    ANIMATED_STORY_TOTAL_SECONDS = int(10 * SECONDS_IN_MINUTE)  # 600 s

    def test_total_video_seconds(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.total_video_seconds == self.ANIMATED_STORY_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.ANIMATED_STORY_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert ANIMATED_STORY_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.ANIMATED_STORY_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert ANIMATED_STORY_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(ANIMATED_STORY_WORKFLOW.model_work.keys()) == expected_models

    def test_model_work_singleton_values(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.GEMMA] == 1
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.FLUX] == 1
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_model_work_subscene_values(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.HF] == ANIMATED_STORY_WORKFLOW.total_subscenes
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.FT] == ANIMATED_STORY_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.HF_VAE] == self.ANIMATED_STORY_TOTAL_SECONDS * FPS[Model.HF]
        assert ANIMATED_STORY_WORKFLOW.model_work[Model.UPSCALER] == self.ANIMATED_STORY_TOTAL_SECONDS * FPS[Model.FT]

    def test_model_work_matches_podcast(self) -> None:
        """AnimatedStory has identical model_work to Podcast."""
        assert ANIMATED_STORY_WORKFLOW.model_work == PODCAST_WORKFLOW.model_work

    def test_hf_num_steps_override(self) -> None:
        """HF denoising steps should be 5% higher than base for LoRA overhead."""
        assert ANIMATED_STORY_WORKFLOW.num_steps[Model.HF] == int(NUM_STEPS[Model.HF] * 1.05)

    def test_other_num_steps_unchanged(self) -> None:
        """Non-HF num_steps should remain at their default values."""
        assert ANIMATED_STORY_WORKFLOW.num_steps[Model.FLUX] == NUM_STEPS[Model.FLUX]
        assert ANIMATED_STORY_WORKFLOW.num_steps[Model.FT] == NUM_STEPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(ANIMATED_STORY_WORKFLOW.models) == {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_parallelizable_models(self) -> None:
        assert ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.HF)
        assert ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.FT)
        assert ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.UPSCALER)
        assert not ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.FLUX)
        assert not ANIMATED_STORY_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.ANIMATED_STORY_TOTAL_SECONDS
        num_scenes = _get_num_scenes(secs)
        num_subscenes = _get_num_subscenes(secs)
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=_video_gen_work(
                secs,
                num_scenes,
                num_subscenes,
                model_work_overrides={Model.FLUX: 1}
            ),
            num_steps_override={Model.HF: int(NUM_STEPS[Model.HF] * 1.05)},
        )
        assert ANIMATED_STORY_WORKFLOW == fresh


class TestLectureWorkflow:
    """Lecture workflow config tests."""
    LECTURE_TOTAL_SECONDS = int(5 * SECONDS_IN_MINUTE)  # 300 s

    def test_total_video_seconds(self) -> None:
        assert LECTURE_WORKFLOW.total_video_seconds == self.LECTURE_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert LECTURE_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.LECTURE_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert LECTURE_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.LECTURE_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert LECTURE_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(LECTURE_WORKFLOW.model_work.keys()) == expected_models

    def test_model_work_singleton_values(self) -> None:
        assert LECTURE_WORKFLOW.model_work[Model.GEMMA] == 1
        assert LECTURE_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_model_work_flux_per_scene(self) -> None:
        """Lecture generates one FLUX image per scene (like Movie)."""
        assert LECTURE_WORKFLOW.model_work[Model.FLUX] == LECTURE_WORKFLOW.total_scenes

    def test_model_work_subscene_values(self) -> None:
        assert LECTURE_WORKFLOW.model_work[Model.HF] == LECTURE_WORKFLOW.total_subscenes
        assert LECTURE_WORKFLOW.model_work[Model.FT] == LECTURE_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert LECTURE_WORKFLOW.model_work[Model.HF_VAE] == self.LECTURE_TOTAL_SECONDS * FPS[Model.HF]
        assert LECTURE_WORKFLOW.model_work[Model.UPSCALER] == self.LECTURE_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(LECTURE_WORKFLOW.models) == {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_flux_parallelizable(self) -> None:
        """Lecture has FLUX work == num_scenes (>1), so it's parallelizable."""
        assert LECTURE_WORKFLOW.is_parallelizable(Model.FLUX)

    def test_parallelizable_models(self) -> None:
        assert LECTURE_WORKFLOW.is_parallelizable(Model.HF)
        assert LECTURE_WORKFLOW.is_parallelizable(Model.FT)
        assert LECTURE_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert LECTURE_WORKFLOW.is_parallelizable(Model.UPSCALER)
        assert LECTURE_WORKFLOW.is_parallelizable(Model.FLUX)
        assert not LECTURE_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not LECTURE_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.LECTURE_TOTAL_SECONDS
        num_sc = _get_num_scenes(secs)
        num_ss = _get_num_subscenes(secs)
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=_video_gen_work(
                secs,
                num_sc,
                num_ss,
                model_work_overrides={Model.FLUX: "num_scenes"},
            ),
        )
        assert LECTURE_WORKFLOW == fresh

    def test_shorter_than_podcast(self) -> None:
        """Lecture is 5 min vs Podcast's 10 min."""
        assert LECTURE_WORKFLOW.total_video_seconds < PODCAST_WORKFLOW.total_video_seconds
        assert LECTURE_WORKFLOW.total_video_seconds == PODCAST_WORKFLOW.total_video_seconds // 2

    def test_fewer_subscenes_than_podcast(self) -> None:
        assert LECTURE_WORKFLOW.total_subscenes < PODCAST_WORKFLOW.total_subscenes

    def test_more_flux_work_than_podcast(self) -> None:
        """Lecture needs FLUX per scene; Podcast only needs 1."""
        assert LECTURE_WORKFLOW.model_work[Model.FLUX] > PODCAST_WORKFLOW.model_work[Model.FLUX]


class TestDubbingWorkflow:
    """Dubbing workflow config tests."""

    DUBBING_TOTAL_SECONDS = int(10 * SECONDS_IN_MINUTE)  # 600 s

    def test_total_video_seconds(self) -> None:
        assert DUBBING_WORKFLOW.total_video_seconds == self.DUBBING_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert DUBBING_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.DUBBING_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert DUBBING_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.DUBBING_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert DUBBING_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(DUBBING_WORKFLOW.model_work.keys()) == expected_models

    def test_no_flux(self) -> None:
        """Dubbing workflow should not include FLUX."""
        assert Model.FLUX not in DUBBING_WORKFLOW.model_work

    def test_model_work_singleton_values(self) -> None:
        assert DUBBING_WORKFLOW.model_work[Model.GEMMA] == 1
        assert DUBBING_WORKFLOW.model_work[Model.OTHERS] == 2

    def test_model_work_subscene_values(self) -> None:
        assert DUBBING_WORKFLOW.model_work[Model.HF] == DUBBING_WORKFLOW.total_subscenes
        assert DUBBING_WORKFLOW.model_work[Model.FT] == DUBBING_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert DUBBING_WORKFLOW.model_work[Model.HF_VAE] == self.DUBBING_TOTAL_SECONDS * FPS[Model.HF]
        assert DUBBING_WORKFLOW.model_work[Model.UPSCALER] == self.DUBBING_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(DUBBING_WORKFLOW.models) == {
            Model.GEMMA,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_parallelizable_models(self) -> None:
        assert DUBBING_WORKFLOW.is_parallelizable(Model.HF)
        assert DUBBING_WORKFLOW.is_parallelizable(Model.FT)
        assert DUBBING_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert DUBBING_WORKFLOW.is_parallelizable(Model.UPSCALER)
        assert DUBBING_WORKFLOW.is_parallelizable(Model.OTHERS)  # work=2 → parallelizable
        assert not DUBBING_WORKFLOW.is_parallelizable(Model.GEMMA)

    def test_others_work_is_two(self) -> None:
        """Dubbing has OTHERS work = 2, unlike Podcast which has 1."""
        assert DUBBING_WORKFLOW.model_work[Model.OTHERS] == 2
        assert DUBBING_WORKFLOW.model_work[Model.OTHERS] > PODCAST_WORKFLOW.model_work[Model.OTHERS]

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.DUBBING_TOTAL_SECONDS
        num_scenes = _get_num_scenes(secs)
        num_ss = _get_num_subscenes(secs)
        work = _video_gen_work(
            secs,
            num_scenes,
            num_ss,
            model_work_overrides={
                Model.FLUX: None,
                Model.OTHERS: 2,
            },
        )
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=work,
        )
        assert DUBBING_WORKFLOW == fresh


class TestEditingWorkflow:
    """Editing workflow config tests."""

    EDITING_TOTAL_SECONDS = int(10 * SECONDS_IN_MINUTE)  # 600 s

    def test_total_video_seconds(self) -> None:
        assert EDITING_WORKFLOW.total_video_seconds == self.EDITING_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert EDITING_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.EDITING_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert EDITING_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.EDITING_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert EDITING_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER,
        }
        assert set(EDITING_WORKFLOW.model_work.keys()) == expected_models

    def test_no_gemma(self) -> None:
        """Editing workflow should not include GEMMA."""
        assert Model.GEMMA not in EDITING_WORKFLOW.model_work

    def test_no_flux(self) -> None:
        """Editing workflow should not include FLUX."""
        assert Model.FLUX not in EDITING_WORKFLOW.model_work

    def test_no_others(self) -> None:
        """Editing workflow should not include OTHERS."""
        assert Model.OTHERS not in EDITING_WORKFLOW.model_work

    def test_model_work_subscene_values(self) -> None:
        assert EDITING_WORKFLOW.model_work[Model.HF] == EDITING_WORKFLOW.total_subscenes
        assert EDITING_WORKFLOW.model_work[Model.FT] == EDITING_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert EDITING_WORKFLOW.model_work[Model.HF_VAE] == self.EDITING_TOTAL_SECONDS * FPS[Model.HF]
        assert EDITING_WORKFLOW.model_work[Model.UPSCALER] == self.EDITING_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(EDITING_WORKFLOW.models) == {
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER,
        }

    def test_parallelizable_models(self) -> None:
        assert EDITING_WORKFLOW.is_parallelizable(Model.HF)
        assert EDITING_WORKFLOW.is_parallelizable(Model.HF_VAE)
        assert EDITING_WORKFLOW.is_parallelizable(Model.FT)
        assert EDITING_WORKFLOW.is_parallelizable(Model.FT_VAE)
        assert EDITING_WORKFLOW.is_parallelizable(Model.UPSCALER)

    def test_fewer_models_than_podcast(self) -> None:
        """Editing has fewer model types than Podcast (no GEMMA, FLUX, OTHERS)."""
        assert len(EDITING_WORKFLOW.model_work) < len(PODCAST_WORKFLOW.model_work)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.EDITING_TOTAL_SECONDS
        num_scenes = _get_num_scenes(secs)
        num_ss = _get_num_subscenes(secs)
        work = _video_gen_work(
            secs,
            num_scenes,
            num_ss,
            model_work_overrides={
                Model.GEMMA: 0,
                Model.FLUX: 0,
                Model.OTHERS: 0,
            }
        )
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=work,
        )
        assert EDITING_WORKFLOW == fresh


class TestVideoChatWorkflow:
    """Video Chat workflow config tests."""

    VIDEO_CHAT_TOTAL_SECONDS = 5  # 5 s

    def test_total_video_seconds(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.total_video_seconds == self.VIDEO_CHAT_TOTAL_SECONDS

    def test_input_tokens(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.total_input_tokens == TOTAL_INPUT_TOKENS

    def test_num_subscenes(self) -> None:
        expected = math.ceil(self.VIDEO_CHAT_TOTAL_SECONDS / SUBSCENE_SECONDS)
        assert VIDEO_CHAT_WORKFLOW.total_subscenes == expected

    def test_num_scenes(self) -> None:
        expected_subscenes = math.ceil(self.VIDEO_CHAT_TOTAL_SECONDS / SUBSCENE_SECONDS)
        expected_scenes = math.ceil(expected_subscenes / SUBSCENES_PER_SCENE)
        assert VIDEO_CHAT_WORKFLOW.total_scenes == expected_scenes

    def test_model_work_keys(self) -> None:
        expected_models = {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }
        assert set(VIDEO_CHAT_WORKFLOW.model_work.keys()) == expected_models

    def test_model_work_singleton_values(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.GEMMA] == 1
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.FLUX] == 1
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.OTHERS] == 1

    def test_model_work_subscene_values(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.HF] == VIDEO_CHAT_WORKFLOW.total_subscenes
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.FT] == VIDEO_CHAT_WORKFLOW.total_subscenes

    def test_model_work_frame_values(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.HF_VAE] == self.VIDEO_CHAT_TOTAL_SECONDS * FPS[Model.HF]
        assert VIDEO_CHAT_WORKFLOW.model_work[Model.UPSCALER] == self.VIDEO_CHAT_TOTAL_SECONDS * FPS[Model.FT]

    def test_config_models(self) -> None:
        assert set(VIDEO_CHAT_WORKFLOW.models) == {
            Model.GEMMA, Model.FLUX,
            Model.HF, Model.HF_VAE,
            Model.FT, Model.FT_VAE,
            Model.UPSCALER, Model.OTHERS,
        }

    def test_same_model_types_as_podcast(self) -> None:
        """Video Chat has the same model types as Podcast."""
        assert set(VIDEO_CHAT_WORKFLOW.model_work.keys()) == set(PODCAST_WORKFLOW.model_work.keys())

    def test_much_shorter_than_podcast(self) -> None:
        """Video Chat is 5s vs Podcast's 600s."""
        assert VIDEO_CHAT_WORKFLOW.total_video_seconds == self.VIDEO_CHAT_TOTAL_SECONDS
        assert VIDEO_CHAT_WORKFLOW.total_video_seconds < PODCAST_WORKFLOW.total_video_seconds

    def test_fewer_subscenes_than_podcast(self) -> None:
        assert VIDEO_CHAT_WORKFLOW.total_subscenes < PODCAST_WORKFLOW.total_subscenes

    def test_parallelizable_models(self) -> None:
        # Singleton models are not parallelizable
        assert not VIDEO_CHAT_WORKFLOW.is_parallelizable(Model.GEMMA)
        assert not VIDEO_CHAT_WORKFLOW.is_parallelizable(Model.FLUX)
        assert not VIDEO_CHAT_WORKFLOW.is_parallelizable(Model.OTHERS)

    def test_rebuild_matches(self) -> None:
        """Rebuilding the config with same params produces the same result."""
        secs = self.VIDEO_CHAT_TOTAL_SECONDS
        num_scenes = _get_num_scenes(secs)
        num_subscenes = _get_num_subscenes(secs)
        fresh = build_workflow_config(
            total_video_seconds=secs,
            input_tokens=TOTAL_INPUT_TOKENS,
            model_work=_video_gen_work(
                secs,
                num_scenes,
                num_subscenes,
                model_work_overrides={Model.FLUX: 1}
            ),
        )
        assert VIDEO_CHAT_WORKFLOW == fresh


# ── STREAMWISE vs NAIVE comparison tests ────────────────────────────────────

@pytest.mark.parametrize("workflow_name,workflow", list(WORKFLOWS.items()))
def test_streamwise_better_than_naive(workflow_name: str, workflow: WorkflowConfig) -> None:
    """STREAMWISE policy should achieve lower cost and TTFF than NAIVE for every workflow."""
    latency_data = load_latency_data("simulator/data/")
    num_gpus = {GPUType.A100: 8, GPUType.H100: 8}

    # Run STREAMWISE (greedy solver, default)
    streamwise_allocator = AutoModelAllocator(
        workflow=workflow,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    streamwise_result = streamwise_allocator.allocate(num_gpus=num_gpus)

    # Run NAIVE
    naive_allocator = AutoModelAllocator(
        workflow=workflow,
        latency_data=latency_data,
        policy=NAIVE_POLICY,
    )
    naive_result = naive_allocator.allocate(num_gpus=num_gpus)

    # STREAMWISE should beat NAIVE on both cost and TTFF
    assert streamwise_result.cost < naive_result.cost, (
        f"[{workflow_name}] STREAMWISE cost ({streamwise_result.cost:.2f}) "
        f"should be less than NAIVE cost ({naive_result.cost:.2f})"
    )
    assert streamwise_result.ttff_s < naive_result.ttff_s, (
        f"[{workflow_name}] STREAMWISE TTFF ({streamwise_result.ttff_s:.2f}s) "
        f"should be less than NAIVE TTFF ({naive_result.ttff_s:.2f}s)"
    )
