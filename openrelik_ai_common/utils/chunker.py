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
import math
from typing import Optional, Tuple, Union

FIRST_PROMPT_CHUNK_WRAPPER = """
**This prompt has (Part 1) of a file's content:** \n
**Please analyze this part of the file:*** \n
```\n{chunk}\n```
"""

PROMPT_CHUNK_WRAPPER = """
**This prompt has (Part {chunk_number}) of a file's content:** \n
**You already analyzed previous parts of the file, here is what you reported so far:** \n
```\n{summary}\n```
**Please analyze this part of the file:*** \n
```\n{chunk}\n```
**Include your analysis report of the previous part in your response.**
"""


class TextFileChunker:
    """
    A class to process and chunk large text files for LLM analysis.

    It iteratively summarizes file content using a "summary of summaries" approach
    to fit within the LLM's context window limitations.
    """

    def __init__(
        self,
        prompt: str,
        file_content: str,
        llm: object,
    ):
        """
        Initializes the TextFileChunker.

        Args:
            prompt: The main prompt to be used for processing each chunk.
            file_content: The content of the file to be processed.
            llm: An instance of the LLM provider.
        """
        self.prompt = prompt
        self.file_content = file_content
        self.llm = llm
        self.chat_session = llm.create_chat_session()

    def process_file_content(self) -> Union[str, object]:
        """
        Processes the file content using a chunking and summarization strategy.

        Returns:
            The final summarized output from the LLM.
        """
        # Calculate the first chunk based on initial prompt and wrapper
        first_chunk_wrapper = self._create_first_chunk_wrapper(chunk="")
        chunk, next_offset = self._get_next_chunk(self.prompt, first_chunk_wrapper)

        if next_offset >= len(self.file_content):
            # If the entire file fits within the first chunk, process it directly
            return self.llm.chat(
                prompt=f"{self.prompt}\n{chunk}",
                as_object=True,
                chat_session=self.chat_session,
            )
        else:
            return self._iterative_summarization(chunk, next_offset)

    def _iterative_summarization(
        self, initial_chunk: str, initial_offset: int
    ) -> Union[str, object]:
        """
        Performs iterative summarization of file chunks.

        Args:
            initial_chunk: The first chunk of the file.
            initial_offset: The offset after the first chunk.

        Returns:
            The final summarized output.
        """
        summary = None
        chunk = initial_chunk
        offset = initial_offset
        chunk_number = 1

        while chunk:
            # Prepare the prompt with appropriate wrapper and current chunk
            prompt_chunk_wrapper = self._create_chunk_wrapper(
                chunk_number, chunk, summary
            )
            full_prompt = f"{self.prompt}\n{prompt_chunk_wrapper}"

            # Determine if this is the last chunk to decide the output format
            is_last_chunk = offset >= len(self.file_content)

            # Generate the summary for the current chunk
            summary = self.llm.chat(
                prompt=full_prompt,
                as_object=is_last_chunk,
                chat_session=self.chat_session,
            )

            # Prepare for the next iteration
            chunk_number += 1
            next_chunk_wrapper = self._create_chunk_wrapper(chunk_number, "", summary)
            chunk, offset = self._get_next_chunk(
                self.prompt, next_chunk_wrapper, offset
            )

            # Reset the chat session if not the last chunk to save context space
            if not is_last_chunk:
                self.chat_session = self.llm.create_chat_session()

        return summary

    def _create_first_chunk_wrapper(self, chunk: str) -> str:
        """
        Creates the wrapper for the first chunk prompt.

        Args:
            chunk: The content of the first chunk.

        Returns:
            The formatted prompt wrapper for the first chunk.
        """
        return FIRST_PROMPT_CHUNK_WRAPPER.format(chunk=chunk)

    def _create_chunk_wrapper(
        self, chunk_number: int, chunk: str, summary: Optional[str] = None
    ) -> str:
        """
        Creates the wrapper for subsequent chunk prompts.

        Args:
            chunk_number: The current chunk number.
            chunk: The content of the current chunk.
            summary: The summary from previous chunks (if any).

        Returns:
            The formatted prompt wrapper for the current chunk.
        """
        if chunk_number == 1:
            return self._create_first_chunk_wrapper(chunk)

        return PROMPT_CHUNK_WRAPPER.format(
            chunk_number=chunk_number, summary=summary, chunk=chunk
        )

    def _get_next_chunk(
        self,
        prompt: str,
        prompt_chunk_wrapper: str,
        offset: int = 0,
    ) -> Tuple[Optional[str], int]:
        """
        Determines the next chunk of the file to be processed.

        Args:
            prompt: The main prompt to be used for processing each chunk.
            prompt_chunk_wrapper: The wrapper for the current chunk's prompt.
            offset: The current offset in the file content.

        Returns:
            A tuple containing the next chunk (or None if end of file) and the updated offset.
        """
        if offset >= len(self.file_content):
            return None, offset

        # Calculate available tokens for the chunk
        available_tokens = self._calculate_available_tokens(
            prompt, prompt_chunk_wrapper
        )

        # Estimate the end character index based on available tokens
        estimated_end_char = min(offset + available_tokens * 4, len(self.file_content))

        # Find a suitable breakpoint for a clean chunk break
        breakpoint = self._find_breakpoint(offset, estimated_end_char)

        # Extract the chunk and update the offset
        chunk = self.file_content[offset:breakpoint]
        return chunk, breakpoint

    def _calculate_available_tokens(
        self, prompt: str, prompt_chunk_wrapper: str
    ) -> int:
        """
        Calculates the number of tokens available for the file chunk.

        Args:
            prompt: The main prompt to be used for processing each chunk.
            prompt_chunk_wrapper: The wrapper for the current chunk's prompt.

        Returns:
            The number of tokens available for the chunk.

        Raises:
            ValueError: If the prompt is too long, leaving no space for content.
        """
        max_tokens = self.llm.get_max_input_tokens(self.llm.config.get("model"))
        prompt_tokens = self._calculate_prompt_tokens(prompt, prompt_chunk_wrapper)
        buffer = self._calculate_dynamic_buffer(max_tokens)

        remaining_tokens = max_tokens - prompt_tokens - buffer

        if remaining_tokens <= 0:
            raise ValueError(
                "Prompt is too long. No space left for file content. "
                f"Max tokens: {max_tokens}, prompt tokens: {prompt_tokens}, "
                f"a buffer of at least {buffer} must be provided "
                " between prompt tokens and max tokens count."
            )

        return remaining_tokens

    def _calculate_prompt_tokens(self, prompt: str, prompt_chunk_wrapper: str) -> int:
        """
        Calculates the number of tokens used by the prompt and system instructions.

        Args:
            prompt: The main prompt to be used for processing each chunk.
            prompt_chunk_wrapper: The wrapper for the current chunk's prompt.

        Returns:
            The total number of tokens used by the prompt.
        """
        return self.llm.count_tokens(
            "\n".join(
                [
                    self.llm.config.get("system_instructions", ""),
                    prompt,
                    prompt_chunk_wrapper,
                ]
            )
        ) + math.ceil(self._get_chat_session_approx_length() / 4)

    def _calculate_dynamic_buffer(self, max_tokens: int) -> int:
        """
        Calculates a dynamic buffer size based on the maximum token limit.

        Args:
            max_tokens: The maximum number of tokens allowed by the model.

        Returns:
            The calculated buffer size.
        """
        base_buffer = 30
        length_factor = 0.0001
        return int(base_buffer + (length_factor * max_tokens))

    def _get_chat_session_approx_length(self) -> int:
        """
        Estimates the string length of the chat session.

        Returns:
            The approximate string length of the chat session.
        """
        return len(str(self.chat_session)) if self.chat_session else 0

    def _find_breakpoint(self, start: int, end: int) -> int:
        """
        Finds a suitable breakpoint for chunking within a given range.

        Args:
            start: The starting index for the search.
            end: The ending index for the search.

        Returns:
            The index of the found breakpoint.
        """
        # Ensure we don't go beyond the file's end
        if end >= len(self.file_content):
            return end

        # Define the minimum acceptable chunk size (10% of the estimated chunk)
        min_chunk_size = int((end - start) * 0.1)

        # Search backwards for punctuation or whitespace
        for i in reversed(range(start, end)):
            if self.file_content[i] in [".", ",", "\n", "\r", " "]:
                if (i - start) >= min_chunk_size:
                    return i
                else:
                    return end  # Not enough content before breakpoint

        return end  # No suitable breakpoint found
