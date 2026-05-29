# Copyright 2024 Google LLC
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
"""Tests for GoogleAI LLM provider."""

import os
import unittest
from unittest.mock import MagicMock, patch

from google.genai import types
from openrelik_ai_common.providers.googleai import GoogleAI


class TestGoogleAI(unittest.TestCase):
    """Test the GoogleAI LLM provider class."""

    def setUp(self):
        """Set up env variables and mocks."""
        os.environ["GOOGLEAI_API_KEY"] = "test_api_key"
        os.environ["GOOGLEAI_DEFAULT_MODEL"] = "gemini-1.5-flash"

        # Mock the entire Client class
        self.client_patcher = patch("openrelik_ai_common.providers.googleai.genai.Client")
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

        # Mock client.chats.create
        self.mock_chat = MagicMock()
        self.mock_chat._curated_history = []
        self.mock_client.chats.create.return_value = self.mock_chat

    def tearDown(self):
        """Tear down patches."""
        self.client_patcher.stop()

    def test_init(self):
        """Test GoogleAI initialization."""
        provider = GoogleAI()
        self.mock_client_class.assert_called_once_with(api_key="test_api_key")
        self.mock_client.chats.create.assert_called_once()
        self.assertEqual(provider.chat_session, self.mock_chat)

    def test_generation_config(self):
        """Test generation config property."""
        provider = GoogleAI(
            temperature=0.7,
            top_p_sampling=0.9,
            top_k_sampling=10,
            system_instructions="test instructions"
        )
        config = provider.generation_config
        self.assertIsInstance(config, types.GenerateContentConfig)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.system_instruction, "test instructions")

    def test_count_tokens(self):
        """Test token counting."""
        provider = GoogleAI()
        self.mock_client.models.count_tokens.return_value = MagicMock(total_tokens=42)

        tokens = provider.count_tokens("hello")
        self.assertEqual(tokens, 42)
        self.mock_client.models.count_tokens.assert_called_once_with(
            model="gemini-1.5-flash",
            contents="hello"
        )

    def test_get_max_input_tokens(self):
        """Test getting max input tokens."""
        provider = GoogleAI()
        self.mock_client.models.get.return_value = MagicMock(input_token_limit=1000)

        max_tokens = provider.get_max_input_tokens("gemini-1.5-flash")
        self.assertEqual(max_tokens, 1000)
        self.mock_client.models.get.assert_called_once_with(model="models/gemini-1.5-flash")

    def test_get_max_input_tokens_with_prefix(self):
        """Test getting max input tokens when name already starts with models/."""
        provider = GoogleAI()
        self.mock_client.models.get.return_value = MagicMock(input_token_limit=1000)

        max_tokens = provider.get_max_input_tokens("models/gemini-1.5-flash")
        self.assertEqual(max_tokens, 1000)
        self.mock_client.models.get.assert_called_once_with(model="models/gemini-1.5-flash")

    def test_generate(self):
        """Test content generation."""
        provider = GoogleAI()
        mock_response = MagicMock()
        mock_response.text = "generated response"
        self.mock_client.models.generate_content.return_value = mock_response

        # Test as text
        result = provider.generate("hello")
        self.assertEqual(result, "generated response")

        # Test as object
        result_obj = provider.generate("hello", as_object=True)
        self.assertEqual(result_obj, mock_response)

    def test_chat(self):
        """Test chat interaction."""
        provider = GoogleAI()
        mock_response = MagicMock()
        mock_response.text = "chat response"
        self.mock_chat.send_message.return_value = mock_response

        # Test as text
        result = provider.chat("hello")
        self.assertEqual(result, "chat response")
        self.mock_chat.send_message.assert_called_once_with("hello")

        # Test as object
        result_obj = provider.chat("hello", as_object=True)
        self.assertEqual(result_obj, mock_response)

    def test_ensure_nonempty_chat_parts(self):
        """Test ensure nonempty chat parts."""
        provider = GoogleAI()
        
        # Create history with an empty parts object and a filled parts object
        mock_content_empty = MagicMock()
        mock_content_empty.parts = []
        mock_content_filled = MagicMock()
        mock_content_filled.parts = [types.Part(text="nonempty")]
        
        self.mock_chat._curated_history = [mock_content_empty, mock_content_filled]
        
        provider._ensure_nonempty_chat_parts()
        
        # The empty one should have been patched with 'ack'
        self.assertEqual(len(mock_content_empty.parts), 1)
        self.assertEqual(mock_content_empty.parts[0].text, "ack")
        # The filled one should remain unchanged
        self.assertEqual(len(mock_content_filled.parts), 1)
        self.assertEqual(mock_content_filled.parts[0].text, "nonempty")
