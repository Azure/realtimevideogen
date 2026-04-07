#!/usr/bin/env python3
"""
Unit tests for StreamCast.
"""
import sys
import os
import json

from dataclasses import asdict

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("apps"):
    from scene import SceneSegment


def test_scene_segment() -> None:
    segment = SceneSegment(
        scene_id=1,
        start_frame=10,
        end_frame=30,
        start_sec=10.0,
        end_sec=35.33
    )
    assert segment.duration_sec == 25.33

    assert len(segment.frame_image_paths) == 0
    segment.add_image_path("path/to/image1.png")
    segment.add_image_path("path/to/image2.png")
    assert len(segment.frame_image_paths) == 2

    assert segment.descriptions == []
    segment.add_description("A sunny beach scene.")
    segment.add_description("Waves crashing on the shore.")
    assert len(segment.descriptions) == 2

    assert segment.get_start() == "00:00:10"
    assert segment.get_end() == "00:00:35"

    segment.transcript = "Scene transcript"

    assert str(segment) is not None
    expected_segment_str = \
        "[  10-  30, 10.0-35.3]: Scene transcript... | " \
        "A sunny beach scene.... | " \
        "Waves crashing on the shore.... | 2 images"
    assert str(segment) == expected_segment_str


def test_user_json_serialization() -> None:
    segment = SceneSegment(
        scene_id=2,
        start_frame=100,
        end_frame=200,
        start_sec=100.0,
        end_sec=150.33
    )

    json_str = json.dumps(asdict(segment))
    data = json.loads(json_str)

    assert data == {
        "audio_path": None,
        "descriptions": [],
        "end_frame": 200,
        "end_sec": 150.33,
        "frame_image_paths": [],
        "language": None,
        "scene_id": 2,
        "start_frame": 100,
        "start_sec": 100.0,
        "transcript": None,
        "translation": None,
    }
