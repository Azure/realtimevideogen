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
sys.path.append("wrapper/slidetranscript")


with patch.dict(sys.modules, {
    'fitz': MagicMock(),
    'azure': MagicMock(),
    'azure.identity': MagicMock(),
    'nvidia_smi': MagicMock(),
    'tenacity': MagicMock(),
    'torch': mock_torch,
    'openai': mock_openai,
}):
    from slidetranscript.wrapper_slidetranscript import SlideTranscriptGenerator


@pytest.mark.asyncio
async def test_slide_transcript() -> None:
    model = SlideTranscriptGenerator()
    assert model is not None
    assert model.model_name == "slidetranscript"

    model.init()
    health = model.get_health()
    assert health is not None
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    # with pytest.raises(ValueError):
    args = await model.get_rest_args({})
    assert args is not None
    args = await model.get_rest_args({
        "pptx_base64": "http://example.com/doc.pdf"
    })
    assert args is not None
    assert args["task"] == "slidetranscript"

    await model.warmup()

    with pytest.raises(ValueError):
        await model.generate()

    pptx_texts = [
        "--- SLIDE 1 ---\nIntroduction to AI",
        "--- SLIDE 2 ---\nMachine Learning Basics",
        "--- SLIDE 3 ---\nDeep Learning Overview",
    ]
    pptx_images = [
        "AAAA",
        "AAAA",
        "AAAA",
    ]

    await model.generate(
        pptx_texts=pptx_texts,
        pptx_images=pptx_images)
    with pytest.raises(ValueError, match="must have the same length"):
        await model.generate(
            pptx_texts=pptx_texts,
            pptx_images=pptx_images[0:1])
    """
    with pytest.raises(Exception, match="Failed to download PDF"):
        await model.generate(pdf_url="http://example.com/doc.pdf")
    with pytest.raises(Exception, match="Cannot query LLM for script"):
        await model.generate(pdf_url="https://arxiv.org/pdf/2501.16634")
    """
