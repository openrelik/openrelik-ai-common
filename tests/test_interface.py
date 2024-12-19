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
from unittest.mock import MagicMock
from typing import Callable, Union
import math
from openrelik_ai_common.providers.interface import (
    LLMProvider,
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


class TestLLMProvider(LLMProvider):
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

    def generate(
        self, prompt: str, file_content: str = None, as_object: bool = False
    ) -> Union[str, object]:
        if as_object:
            return {"response": f"Generated: {prompt}"}
        return f"Generated: {prompt}"

    def chat(
        self, prompt: str, file_content: str = None, as_object: bool = False
    ) -> Union[str, object]:
        self.chat_session.append(prompt)
        if as_object:
            return {"response": f"Chat response: {prompt}"}
        return f"Chat response: {prompt}"


class TestDoChunkedPrompt(unittest.TestCase):
    def setUp(self):
        self.provider = TestLLMProvider()

    def test_single_chunk(self):
        prompt = "Test prompt"
        # file content is small enough to fit in one chunk
        file_content = "This is a short file content."
        mock_prompt_function = MagicMock(
            return_value={"response": "Summary of single chunk"}
        )

        result = self.provider.do_chunked_prompt(
            prompt, file_content, mock_prompt_function
        )

        self.assertEqual(result, {"response": "Summary of single chunk"})
        mock_prompt_function.assert_called_once()
        args, kwargs = mock_prompt_function.call_args
        # Assert that args is empty
        # We only pass named args, these should be in kwargs
        self.assertEqual(len(args), 0)
        self.assertEqual(
            kwargs["prompt"],  # Access prompt from kwargs
            f"{prompt}\n{file_content}",
        )
        self.assertEqual(kwargs["as_object"], True)

    def test_prompt_too_long(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=100)
        self.provider.count_tokens = MagicMock(return_value=200)
        file_content = "This is a test file."
        prompt = ""
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk=file_content
        )
        offset = 0

        with self.assertRaisesRegex(
            ValueError,
            "Prompt is too long. No space left for file content. "
            "Max tokens: 100, prompt tokens: 200",
        ):
            self.provider._get_next_chunk(
                file_content, prompt, prompt_chunk_wrapper, offset
            )

    def test_multiple_chunks(self):
        # system instruction and wrapper around 50 tokens, 4 tokens prompt
        prompt = "prompt" * 4
        # 30 chars sentence we need ~ 3000 - 30 * 3 tokens more to make
        # sure we trigger 4 turns
        file_content = "This is a long file content!!!" * 4 * 97
        # Simulate multiple chunks
        mock_prompt_function = MagicMock(
            side_effect=[
                "Summary 1",
                "Summary 2",
                "Summary 3",
                {"response": "Summary 4"},
            ]
        )

        result = self.provider.do_chunked_prompt(
            prompt, file_content, mock_prompt_function
        )

        self.assertEqual(result, {"response": "Summary 4"})
        self.assertEqual(mock_prompt_function.call_count, 4)
        self.assertEqual(self.provider.chat_session, [])

    def test_empty_file(self):
        prompt = "Test prompt"
        file_content = ""
        mock_prompt_function = MagicMock(return_value={"response": "Summary of empty"})

        result = self.provider.do_chunked_prompt(
            prompt, file_content, mock_prompt_function
        )

        self.assertEqual(result, {"response": "Summary of empty"})
        mock_prompt_function.assert_called_once()

        # Check if args is empty before accessing elements
        args, kwargs = mock_prompt_function.call_args
        self.assertEqual(len(args), 0)  # Assert args is empty
        self.assertEqual(
            kwargs,
            {
                "prompt": "Test prompt\nNone",
                "as_object": True,
            },
        )

    def test_get_next_chunk_clean_break(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)

        file_content = "This is a sentence. This is another sentence that is long."
        prompt = "Summarize"
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="This is a sentence. "
        )
        offset = 0

        chunk, next_offset = self.provider._get_next_chunk(
            file_content, prompt, prompt_chunk_wrapper, offset
        )

        self.assertEqual(chunk, "This is a sentence.")
        self.assertEqual(next_offset, 19)

    def test_get_next_chunk_no_clean_break(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)

        file_content = "ThisIsASentenceWithNoSpaces"
        prompt = "Summarize"
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="ThisIsASentenceWithNoSpaces"
        )
        offset = 0

        chunk, next_offset = self.provider._get_next_chunk(
            file_content, prompt, prompt_chunk_wrapper, offset
        )
        self.assertEqual(chunk, "ThisIsASente")
        self.assertEqual(next_offset, 12)

    def test_get_next_chunk_end_of_file(self):
        self.provider.get_max_input_tokens = MagicMock(return_value=73)
        file_content = "Short sentence."
        prompt = "Summarize"
        prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
            i=1, chunk="Short sentence."
        )
        offset = 0

        chunk, next_offset = self.provider._get_next_chunk(
            file_content, prompt, prompt_chunk_wrapper, offset
        )

        self.assertEqual(chunk, "Short sentence.")
        self.assertEqual(next_offset, 15)

        # Call again with offset at end of file
        chunk, next_offset = self.provider._get_next_chunk(
            file_content, prompt, prompt_chunk_wrapper, next_offset
        )
        self.assertIsNone(chunk)
        self.assertEqual(next_offset, 15)

    def test_get_chat_session_approx_string_length(self):
        self.provider.chat_session = ["Message 1", "Message 2"]
        length = self.provider._get_chat_session_approx_string_length()
        self.assertEqual(length, len(str(["Message 1", "Message 2"])))

        self.provider.chat_session = []
        length = self.provider._get_chat_session_approx_string_length()
        self.assertEqual(length, 0)

    def test_find_breakpoint_period(self):
        text = "Thisisasentence.Anothersentence"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_comma(self):
        text = "Thisisasentence,withacomma"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_newline(self):
        text = "Thisisasentence\nAnothersentence"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 15)

    def test_find_breakpoint_space(self):
        text = "Thisisa sentencewithspacebreak"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 7)

    def test_find_breakpoint_no_breakpoint(self):
        text = "ThisIsASentenceWithoutAnyBreakpoints"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, end)

    def test_find_breakpoint_end_at_last_char(self):
        text = "This is a sentence."
        start = 0
        end = len(text)
        breakpoint = self.provider._find_breakpoint(text, start, end)
        # Should return end immediately
        self.assertEqual(breakpoint, end)

    def test_find_breakpoint_empty_text(self):
        text = ""
        start = 0
        end = 0
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 0)

    def test_find_breakpoint_start_equals_end(self):
        text = "This is a sentence."
        start = 10
        end = 10
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, 10)

    def test_find_breakpoint_reversed_range(self):
        text = "This is a sentence. Another sentence."
        start = 0
        end = 20

        # Call the method with a reversed range (start > end)
        breakpoint = self.provider._find_breakpoint(text, end, start)

        # Assert that the method handles it correctly, returning the end
        self.assertEqual(breakpoint, start)

    def test_find_breakpoint_less_than_10_percent(self):
        text = "Th.isisaverylongsentencewithonlyoneseparatoratthebeginning"
        start = 0
        end = len(text) - 1
        breakpoint = self.provider._find_breakpoint(text, start, end)
        self.assertEqual(breakpoint, end)


if __name__ == "__main__":
    unittest.main()
