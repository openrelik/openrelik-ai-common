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
from typing import Callable, Tuple, Union, Optional

from .config import get_provider_config

# Default values
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P_SAMPLING = 0.1
DEFAULT_TOP_K_SAMPLING = 1

FIRST_PROMPT_CHUNK_WRAPPER = """
**This prompt has (Part {i}) of a file content:** \n
**Please analyze this pars of the file:*** \n
```\n{chunk}\n```
"""

PROMPT_CHUNK_WRAPPER = """
**This prompt has (Part {i}) of a file content:** \n
**You already analyzed previous parts of the file, here was your report so far:** \n
```\n{summary}\n```
**Please analyze this pars of the file:*** \n
```\n{chunk}\n```
"""


class LLMProvider:
  """Base class for LLM providers."""

  NAME = "name"
  DISPLAY_NAME = "display_name"

  def __init__(
      self,
      model_name: str = None,
      system_instructions: str = None,
      temperature: float = DEFAULT_TEMPERATURE,
      top_p_sampling: float = DEFAULT_TOP_P_SAMPLING,
      top_k_sampling: int = DEFAULT_TOP_K_SAMPLING,
      max_input_tokens: int = None,
  ):
    """Initialize the LLM provider.

    Args:
        model_name: The name of the model to use.
        system_instructions: The system instruction to use for the response.
        temperature: The temperature to use for the response.
        top_p_sampling: The top_p sampling to use for the response.
        top_k_sampling: The top_k sampling to use for the response.
        max_input_tokens: The maximum number of input tokens per prompt, default
          is set using max_input_tokens.

    Attributes:
        config: The configuration for the LLM provider.

    Raises:
        Exception: If the LLM provider is not configured.
    """
    config = {}
    config["model"] = model_name
    config["system_instructions"] = system_instructions
    config["temperature"] = temperature
    config["top_p_sampling"] = top_p_sampling
    config["top_k_sampling"] = top_k_sampling

    # Load the LLM provider config from environment variables.
    config_from_environment = get_provider_config(self.NAME)
    if not config_from_environment:
      raise ValueError(f"{self.NAME} config not found")
    config.update(config_from_environment)

    if not model_name:
      config["model"] = config.get("default_model")

    # Expose the config as an attribute.
    self.config = config

    # Create chat session.
    self.chat_session = None

    # Set the max input tokens count
    self.max_input_tokens = max_input_tokens

  def to_dict(self):
    """Convert the LLM provider to a dictionary.

    Returns:
        A dictionary representation of the LLM provider.
    """
    return {
        "name": self.NAME,
        "display_name": self.DISPLAY_NAME,
        "config": {
            "model": self.config.get("model"),
            "system_instructions": self.config.get("system_instructions"),
            "temperature": self.config.get("temperature"),
            "top_p_sampling": self.config.get("top_p_sampling"),
            "top_k_sampling": self.config.get("top_k_sampling"),
        },
    }

  @property
  def generation_config(self):
    """Get the generation config for the LLM provider.

    Returns:
        A dictionary representation of the generation config.
    """
    return NotImplementedError()

  def create_chat_session(self):
    """Create chat session object.

    Returns:
        Chat session.
    """
    raise NotImplementedError()

  def count_tokens(self, prompt: str):
    """Count the number of tokens in a prompt.

    Args:
        prompt: The prompt to count the tokens for.

    Returns:
        The number of tokens in the prompt.
    """
    raise NotImplementedError()

  def max_input_tokens(self, model_name: str):
    """Get the max number of input tokens allowed for a model.

    Args:
        model_name: Model name to get max input token number for.

    Returns:
        The max number of input tokens allowed.
    """
    raise NotImplementedError()

  def generate(
      self, prompt: str, file_content: str = None, as_object: bool = False
  ) -> Union[str, object]:
    """Generate a response from the LLM provider.

    Args:
        prompt: The prompt to generate a response for.
        file_content: If file_content is provided and the overall prompt limit
          is more than maximum allowed input token count, then the file content
          will be split into chunks and iterative summary will be returned and
          used in the history session.
        as_object: return response object from API else text.

    Returns:
        The generated response.
    """
    raise NotImplementedError()

  def chat(
      self, prompt: str, file_content: str = None, as_object: bool = False
  ) -> Union[str, object]:
    """Chat using the LLM provider.

    Args:
        prompt: The user prompt to chat with.
        file_content: If file_content is provided and the overall prompt limit
          is more than maximum allowed input token count, then the file content
          will be split into chunks and iterative summary will be returned and
          used in the history session.
        as_object: return response object from API else text.

    Returns:
        The chat response.
    """
    raise NotImplementedError()

  def do_chunked_prompt(
      self,
      prompt: str,
      file_content: str,
      prompt_function: Callable[..., Union[str, object]],
  ) -> Union[str, object]:
    """Do a chunked prompt.

    Args:
        prompt: The prompt to generate a response for.
        file_content: The file content to chunk.
        prompt_function: The function to call to generate the response.

    Returns:
        The generated response.
    """
    chunk, offset = self._get_next_chunk(
        file_content, prompt, FIRST_PROMPT_CHUNK_WRAPPER
    )
    summary = None
    if offset < len(file_content):
      # The data fits in single prompt
      summary = prompt_function(prompt=f"{prompt}\n{chunk}", as_object=True)
    else:
      chunk_number = 1
      while chunk:
        if chunk_number == 1:
          prompt_chunk_wrapper = FIRST_PROMPT_CHUNK_WRAPPER.format(
              i=chunk_number, chunk=chunk
          )
        else:
          prompt_chunk_wrapper = PROMPT_CHUNK_WRAPPER.format(
              i=chunk_number, chunk=chunk, summary=summary
          )
        # Make response always text except if last part
        summary = prompt_function(
            prompt=f"{prompt}\n{prompt_chunk_wrapper}",
            as_object=False if offset < len(file_content) else True,
        )
        chunk_number += 1
        chunk, offset = self._get_next_chunk(
            file_content,
            prompt,
            PROMPT_CHUNK_WRAPPER.format(
                i=chunk_number, chunk=chunk, summary=summary
            ),
            offset,
            summary,
        )
        # if this chunk is not the last
        if offset < len(file_content):
          # The latest summary will be included in the new prompt,
          # clearing history as it is counted into the context window.
          self.chat_session = self.create_chat_session()
    return summary

  def _get_next_chunk(
      self,
      file_content: str,
      prompt: str,
      prompt_chunk_wrapper: str,
      offset: int = 0,
      summary: str = None,
  ) -> Tuple[Optional[str], int]:
    """Chunks a string into segments of a maximum estimated token size.

    Assumes an average token length of 4 characters.

    Args:
        file_content: The input string.
        prompt: The prompt to generate a response for.
        prompt_chunk_wrapper: The wrapper to use for the prompt chunk.
        offset: The offset to start chunking from.
        summary: The summary of the previous chunks.

    Returns:
        A list of strings (chunks).
    """
    max_size = self.max_input_tokens(self, self.config.get("model"))
    prompt_token_count = self.count_tokens(
        "\n".join([
            self.config.get("system_instructions", ""),
            prompt,
            prompt_chunk_wrapper,
            summary or "",
        ])
    ) + math.ceil(self._get_chat_session_approx_string_length() / 4)
    # Subtracting 100 chars as buffer to cover for indentations
    # or inaccuracies due to assuming that a token is 4 chars.
    remaning_tokens = max_size - prompt_token_count - 100
    chunk = None
    if offset < len(file_content):
      end_char = min(
          offset + remaning_tokens * 4, len(file_content)
      )  # Estimate end char index

      # Try to find a space to break the chunk more cleanly
      break_point = file_content.rfind(" ", offset, end_char)
      if break_point == -1:
        break_point = end_char

      chunk = file_content[offset:break_point]
      offset = break_point + 1
    return chunk, offset

  def _get_chat_session_approx_string_length(self):
    """Calculates string length of the chat session object.

    ToDo: overrides this function in providers classes to provide more
    accurate representation of chat session length.

    Returns:
        The length of the string representation of the object.
    """
    return len(str(self.chat_session)) if self.chat_session else 0
