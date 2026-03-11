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
