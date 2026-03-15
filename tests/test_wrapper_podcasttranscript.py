#!/usr/bin/env python3

import sys
import pytest
from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.openaiclient_mock import OpenAIClientMock

mock_torch = TorchMock()
mock_openai = OpenAIClientMock()

sys.path.append("wrapper")
sys.path.append("wrapper/podcasttranscript")

with patch.dict(sys.modules, {
    'fitz': MagicMock(),
    'azure': MagicMock(),
    'azure.identity': MagicMock(),
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'tenacity': MagicMock(),
    'torch': mock_torch,
    'openai': mock_openai,
}):
    from podcasttranscript.wrapper_podcasttranscript import PodcastTranscriptGenerator
    from podcasttranscript.wrapper_podcasttranscript import Dialogue
    from podcasttranscript.wrapper_podcasttranscript import Script
    from podcasttranscript.wrapper_podcasttranscript import Scene
    from podcasttranscript.wrapper_podcasttranscript import Podcast


@pytest.mark.asyncio
async def test_podcast_transcript() -> None:
    model = PodcastTranscriptGenerator()
    assert model is not None
    assert model.model_name == "podcasttranscript"

    model.init()
    health = model.get_health()
    assert health is not None
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "pdf_url": "http://example.com/doc.pdf"
    })

    await model.warmup()

    with pytest.raises(ValueError):
        await model.generate()
    with pytest.raises(Exception, match="Failed to download PDF"):
        await model.generate(pdf_url="http://example.com/doc.pdf")
    with pytest.raises(Exception, match="Cannot query LLM for script"):
        await model.generate(pdf_url="https://arxiv.org/pdf/2501.16634")


def test_dialogue_str() -> None:
    """Dialogue.__str__() should format as 'character: transcript'."""
    d = Dialogue(character="Alice", transcript="Hello world.")
    assert str(d) == "Alice: Hello world."

    d_end = Dialogue(character="Bob", transcript="Goodbye.", end_script=True)
    assert str(d_end) == "Bob: Goodbye."


def test_script_str() -> None:
    """Script.__str__() should list all dialogues prefixed with 'Script:'."""
    script = Script(dialogues=[
        Dialogue(character="Alice", transcript="Hi there."),
        Dialogue(character="Bob", transcript="Hey!"),
    ])
    result = str(script)
    assert result.startswith("Script:")
    assert "Alice: Hi there." in result
    assert "Bob: Hey!" in result


def test_scene_str() -> None:
    """Scene.__str__() should list characters and all dialogue lines."""
    scene = Scene(
        characters=["Alice", "Bob"],
        dialogues=[
            Dialogue(character="Alice", transcript="Welcome."),
            Dialogue(character="Bob", transcript="Thanks."),
        ],
    )
    result = str(scene)
    assert "Alice" in result
    assert "Bob" in result
    assert "Welcome." in result
    assert "Thanks." in result


def test_podcast_str() -> None:
    """Podcast.__str__() should include scene indices and dialogue text."""
    podcast = Podcast(scenes=[
        Scene(
            characters=["Alice"],
            dialogues=[Dialogue(character="Alice", transcript="Scene zero.")],
        ),
        Scene(
            characters=["Bob"],
            dialogues=[Dialogue(character="Bob", transcript="Scene one.")],
        ),
    ])
    result = str(podcast)
    assert "Scene: 0" in result
    assert "Scene: 1" in result
    assert "Scene zero." in result
    assert "Scene one." in result
