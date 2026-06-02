# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Anthropic Claude LLM providers."""

import os
import unittest
from unittest.mock import MagicMock, patch

from openrelik_ai_common.providers.anthropic import (
    DEFAULT_CONTEXT_WINDOW_SIZE,
    AnthropicProvider,
)
from openrelik_ai_common.providers.anthropicvertex import AnthropicVertexProvider


def _mock_response(text="mock response"):
    """Build a mock Messages API response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]
    return response


class TestAnthropicProvider(unittest.TestCase):
    """Test the Anthropic Claude LLM provider class."""

    def setUp(self):
        """Set up env variables and mocks."""
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"
        os.environ["ANTHROPIC_DEFAULT_MODEL"] = "claude-sonnet-4-6"

        self.client_patcher = patch(
            "openrelik_ai_common.providers.anthropic.Anthropic"
        )
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

    def tearDown(self):
        """Tear down patches."""
        self.client_patcher.stop()

    def test_init(self):
        """Test provider initialization."""
        provider = AnthropicProvider()
        self.mock_client_class.assert_called_once_with(api_key="test_api_key")
        self.assertEqual(provider.config.get("model"), "claude-sonnet-4-6")
        self.assertIsNotNone(provider.chat_session)

    def test_generation_config(self):
        """Test generation config property."""
        provider = AnthropicProvider(
            temperature=0.7, top_p_sampling=0.9, top_k_sampling=10
        )
        config = provider.generation_config
        self.assertEqual(config["temperature"], 0.7)
        # Claude 4 models accept a single sampling control per request, so
        # only temperature is forwarded.
        self.assertNotIn("top_p", config)
        self.assertNotIn("top_k", config)

    def test_generate(self):
        """Test text generation."""
        self.mock_client.messages.create.return_value = _mock_response("generated")
        provider = AnthropicProvider()

        result = provider.generate("test prompt")

        self.assertEqual(result, "generated")
        _, kwargs = self.mock_client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "test prompt"}])
        self.assertNotIn("system", kwargs)

    def test_generate_with_system_instructions(self):
        """Test that system instructions are sent with prompt caching enabled."""
        self.mock_client.messages.create.return_value = _mock_response()
        provider = AnthropicProvider(system_instructions="be concise")

        provider.generate("test prompt")

        _, kwargs = self.mock_client.messages.create.call_args
        self.assertEqual(kwargs["system"][0]["text"], "be concise")
        self.assertEqual(kwargs["system"][0]["cache_control"], {"type": "ephemeral"})

    def test_generate_as_object(self):
        """Test generation returning the raw response object."""
        response = _mock_response()
        self.mock_client.messages.create.return_value = response
        provider = AnthropicProvider()

        self.assertEqual(provider.generate("test prompt", as_object=True), response)

    def test_chat(self):
        """Test multi-turn chat keeps history."""
        self.mock_client.messages.create.side_effect = [
            _mock_response("first"),
            _mock_response("second"),
        ]
        provider = AnthropicProvider()

        first = provider.chat("hello")
        second = provider.chat("again")

        self.assertEqual(first, "first")
        self.assertEqual(second, "second")
        _, kwargs = self.mock_client.messages.create.call_args
        # history: user, assistant, user
        self.assertEqual(len(kwargs["messages"]), 3)
        self.assertEqual(kwargs["messages"][1]["content"], "first")

    def test_count_tokens(self):
        """Test token counting via the API."""
        count_response = MagicMock()
        count_response.input_tokens = 42
        self.mock_client.messages.count_tokens.return_value = count_response
        provider = AnthropicProvider()

        self.assertEqual(provider.count_tokens("test prompt"), 42)

    def test_count_tokens_fallback(self):
        """Test token counting falls back to an estimate on API errors."""
        from anthropic import APIConnectionError

        self.mock_client.messages.count_tokens.side_effect = APIConnectionError(
            request=MagicMock()
        )
        provider = AnthropicProvider()

        prompt = "x" * 300
        self.assertEqual(provider.count_tokens(prompt), 100)

    def test_get_max_input_tokens(self):
        """Test default context window size."""
        provider = AnthropicProvider()
        self.assertEqual(provider.get_max_input_tokens(), DEFAULT_CONTEXT_WINDOW_SIZE)

    def test_get_max_input_tokens_override(self):
        """Test explicit max input tokens override."""
        provider = AnthropicProvider(max_input_tokens=1000)
        self.assertEqual(provider.get_max_input_tokens(), 1000)

    def test_response_to_text(self):
        """Test response object to text conversion."""
        provider = AnthropicProvider()
        self.assertEqual(provider.response_to_text(_mock_response("text")), "text")


class TestAnthropicVertexProvider(unittest.TestCase):
    """Test the Anthropic Claude on Vertex AI provider class."""

    def setUp(self):
        """Set up env variables and mocks."""
        os.environ["ANTHROPICVERTEX_PROJECT_ID"] = "test-project"
        os.environ["ANTHROPICVERTEX_REGION"] = "us-east5"
        os.environ["ANTHROPICVERTEX_DEFAULT_MODEL"] = "claude-sonnet-4-6"

        self.client_patcher = patch(
            "openrelik_ai_common.providers.anthropicvertex.AnthropicVertex"
        )
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

    def tearDown(self):
        """Tear down patches."""
        self.client_patcher.stop()

    def test_init_uses_adc_not_api_key(self):
        """Test the Vertex client is created from project/region (ADC auth)."""
        provider = AnthropicVertexProvider()
        self.mock_client_class.assert_called_once_with(
            project_id="test-project", region="us-east5"
        )
        self.assertEqual(provider.config.get("model"), "claude-sonnet-4-6")

    def test_generate(self):
        """Test text generation via the Vertex client."""
        self.mock_client.messages.create.return_value = _mock_response("from vertex")
        provider = AnthropicVertexProvider()

        self.assertEqual(provider.generate("test prompt"), "from vertex")


if __name__ == "__main__":
    unittest.main()
