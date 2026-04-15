"""
# OpenAI client mock.
"""

from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from typing import Dict
from typing import Any

mock_openai = MagicMock()
mock_client = MagicMock()
mock_client.chat.completions.create = AsyncMock()
mock_openai.AsyncOpenAI.return_value = mock_client


class OpenAIClientMock:

    def __init__(self) -> None:
        self.response = MagicMock()
        self.response.choices = [
            MagicMock(message=MagicMock(content="mock response"))
        ]
        self.client = MagicMock()
        self.client.chat.completions.create = AsyncMock(
            return_value=self.response
        )

        self.AsyncOpenAI = MagicMock(return_value=self.client)
        self.AsyncAzureOpenAI = MagicMock(return_value=self.client)

    def get_sub_modules(self) -> Dict[str, Any]:
        openai_module = MagicMock()
        openai_module.AsyncOpenAI = self.AsyncOpenAI
        openai_module.AsyncAzureOpenAI = self.AsyncAzureOpenAI
        return {
            "openai": openai_module,
        }
