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

import os
import unittest
from typing import Union
from unittest.mock import MagicMock

from openrelik_ai_common.providers.interface import (
    LLMProvider,
)
from openrelik_ai_common.utils.chunker import (
    TextFileChunker,
    FIRST_PROMPT_CHUNK_WRAPPER,
)


# Dummy config for testing
TEST_CONFIG = {
    "model": "test_model",
    "system_instructions": "test instructions",
    "temperature": 0.5,
    "top_p_sampling": 0.5,
    "top_k_sampling": 5,
    "default_model": "test_model",
}


class MockLLMProvider(LLMProvider):
    NAME = "googleai"
    DISPLAY_NAME = "Test Provider"

    def __init__(self, model_name: str = None):
        # Set dummy environment variables for testing
        os.environ["GOOGLEAI_API_KEY"] = "dummy_api_key"
        os.environ["GOOGLEAI_DEFAULT_MODEL"] = "dummy_model"
        super().__init__(model_name=model_name)
        self.config = TEST_CONFIG
        self.chat_session = []

    def create_chat_session(self):
        self.chat_session = []
        return self.chat_session

    def count_tokens(self, prompt: str):
        return len(prompt) // 4

    def get_max_input_tokens(self, model_name: str):
        return 1000

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        if as_object:
            return {"response": f"Generated: {prompt}"}
        return f"Generated: {prompt}"

    def chat(
        self, prompt: str, as_object: bool = False, chat_session: object = None
    ) -> Union[str, object]:
        self.chat_session.append(prompt)
        if as_object:
            return {"response": f"Chat response: {prompt}"}
        return f"Chat response: {prompt}"


class TestDoChunkedPrompt(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_single_chunk(self):
        prompt = "Test prompt"
        # file content is small enough to fit in one chunk
        file_content = "This is a short file content."
        mock_prompt_function = MagicMock(
            return_value={"response": "Summary of single chunk"}
        )

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )

        result = chunker.process_file_content()

        self.assertEqual(
            result,
            {"response": "Chat response: Test prompt\nThis is a short file content."},
        )

    def test_prompt_too_long(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=100)
        self.provider.count_tokens = MagicMock(return_value=200)
        prompt = ""
        file_content = "This is a test file."
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk=file_content
        )
        offset = 0

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Prompt is too long. No space left for file content. "
            "Max tokens: 100, prompt tokens: 200",
        ):
            chunker._get_next_chunk(prompt, prompt_chunk_wrapper, offset)

    def test_empty_file(self):
        prompt = "Test prompt"
        file_content = ""

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )
        result = chunker.process_file_content()
        self.assertEqual(result, {"response": "Chat response: Test prompt\nNone"})

    def test_get_next_chunk_clean_break(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)

        prompt = "Summarize"
        file_content = "This is a sentence. This is another sentence that is long."
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="This is a sentence. "
        )
        offset = 0

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )

        chunk, next_offset = chunker._get_next_chunk(
            prompt, prompt_chunk_wrapper, offset
        )

        self.assertEqual(chunk, "This is a sentence.")
        self.assertEqual(next_offset, 19)

    def test_get_next_chunk_no_clean_break(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)

        prompt = "Summarize"
        file_content = "ThisIsASentenceWithNoSpaces"
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="ThisIsASentenceWithNoSpaces"
        )
        offset = 0

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )

        chunk, next_offset = chunker._get_next_chunk(
            prompt, prompt_chunk_wrapper, offset
        )
        self.assertEqual(chunk, "ThisIsASente")
        self.assertEqual(next_offset, 12)

    def test_get_next_chunk_end_of_file(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)

        prompt = "Summarize"
        file_content = "Short sentence."
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="Short sentence."
        )
        offset = 0

        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self.provider,
        )

        chunk, next_offset = chunker._get_next_chunk(
            prompt, prompt_chunk_wrapper, offset
        )

        self.assertEqual(chunk, "Short sentence.")
        self.assertEqual(next_offset, 15)

        # Call again with offset at end of file
        chunk, next_offset = chunker._get_next_chunk(
            prompt, prompt_chunk_wrapper, next_offset
        )
        self.assertIsNone(chunk)
        self.assertEqual(next_offset, 15)

    def test_get_chat_session_approx_length(self):
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content="Test file content",
            llm=self.provider,
        )
        chunker.chat_session = ["Message 1", "Message 2"]
        length = chunker._get_chat_session_approx_length()
        self.assertEqual(length, len(str(["Message 1", "Message 2"])))

        chunker.chat_session = []
        length = chunker._get_chat_session_approx_length()
        self.assertEqual(length, 0)

    def test_find_breakpoint_period(self):
        file_content = "Thisisasentence.Anothersentence"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_comma(self):
        file_content = "Thisisasentence,withacomma"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_newline(self):
        file_content = "Thisisasentence\nAnothersentence"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_space(self):
        file_content = "Thisisa sentencewithspacebreak"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 7)

    def test_find_breakpoint_no_breakpoint(self):
        file_content = "ThisIsASentenceWithoutAnyBreakpoints"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, end)

    def test_find_breakpoint_end_at_last_char(self):
        file_content = "This is a sentence."
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content)
        breakpoint = chunker._find_breakpoint(start, end)
        # Should return end immediately
        self.assertEqual(breakpoint, end)

    def test_find_breakpoint_empty_text(self):
        file_content = ""
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = 0
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 0)

    def test_find_breakpoint_start_equals_end(self):
        file_content = "This is a sentence."
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 10
        end = 10
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, 10)

    def test_find_breakpoint_reversed_range(self):
        file_content = "This is a sentence. Another sentence."
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = 20

        # Call the method with a reversed range (start > end)
        breakpoint = chunker._find_breakpoint(end, start)

        # Assert that the method handles it correctly, returning the end
        self.assertEqual(breakpoint, start)

    def test_find_breakpoint_less_than_10_percent(self):
        file_content = "Th.isisaverylongsentencewithonlyoneseparatoratthebeginning"
        chunker = TextFileChunker(
            prompt="Test prompt",
            file_content=file_content,
            llm=self.provider,
        )
        start = 0
        end = len(file_content) - 1
        breakpoint = chunker._find_breakpoint(start, end)
        self.assertEqual(breakpoint, end)


if __name__ == "__main__":
    unittest.main()
