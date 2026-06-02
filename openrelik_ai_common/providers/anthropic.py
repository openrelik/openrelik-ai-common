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

"""Anthropic Claude LLM provider."""

import logging
import os
from typing import Union

import backoff
from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError

from . import interface, manager

ONE_MINUTE = 60  # One minute in seconds
TEN_MINUTES = 10 * ONE_MINUTE

# The Messages API requires an explicit output token limit.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

# Context window sizes for Claude models. Models not listed fall back to the
# default, which is correct for all current generations.
MODEL_CONTEXT_WINDOW_SIZE = {}
DEFAULT_CONTEXT_WINDOW_SIZE = 200000

logger = logging.getLogger(__name__)


def _backoff_handler(details) -> None:
    """Backoff handler for Anthropic API calls."""
    logger.info("Backing off %s seconds after %s tries", details["wait"], details["tries"])


class ChatSession:
    """A chat session with the Anthropic Messages API."""

    def __init__(
        self,
        client: Anthropic,
        model_name: str,
        generation_config: dict,
        system: str = None,
    ):
        self.client = client
        self.model = model_name
        self.generation_config = generation_config
        self.system = system
        self.history = []

    def send_message(self, prompt: str):
        """Send a message to the chat session.

        Args:
            prompt: The user message to send.

        Returns:
            The response from the Anthropic API.
        """
        self.history.append({"role": "user", "content": prompt})
        kwargs = {}
        if self.system:
            kwargs["system"] = self.system
        response = self.client.messages.create(
            model=self.model,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            # Snapshot so later history updates can't alias into the request.
            messages=list(self.history),
            **self.generation_config,
            **kwargs,
        )
        self.history.append({"role": "assistant", "content": response.content[0].text})
        return response


class AnthropicProvider(interface.LLMProvider):
    """Anthropic Claude LLM provider (api.anthropic.com, API key auth)."""

    NAME = "anthropic"
    DISPLAY_NAME = "Anthropic Claude"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = self._create_client()
        self.chat_session = self.create_chat_session()

    def _create_client(self):
        """Create the API client. Subclasses override the auth mechanism."""
        return Anthropic(api_key=self.config.get("api_key"))

    @property
    def generation_config(self) -> dict:
        """Get the generation config for the Anthropic provider.

        Claude 4 models accept only one sampling control per request, so
        temperature is used and nucleus/top-k sampling are not forwarded.
        """
        return {
            "temperature": self.config.get("temperature"),
        }

    def _system_blocks(self):
        """System instructions as content blocks with prompt caching enabled.

        The cache_control marker lets repeated calls that share the same system
        instructions (e.g. per-chunk analysis of a large file) reuse the cached
        prefix instead of reprocessing it.
        """
        system = self.config.get("system_instructions")
        if not system:
            return None
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def create_chat_session(self) -> ChatSession:
        """Create a chat session with the Anthropic API."""
        return ChatSession(
            client=self.client,
            model_name=self.config.get("model"),
            generation_config=self.generation_config,
            system=self._system_blocks(),
        )

    def count_tokens(self, prompt: str) -> int:
        """Count the number of tokens in a prompt using the Anthropic API.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            int: The number of tokens in the prompt.
        """
        try:
            response = self.client.messages.count_tokens(
                model=self.config.get("model"),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.input_tokens
        except (APIStatusError, APIConnectionError):
            # Conservative estimate if the counting endpoint is unavailable.
            return len(prompt) // 3

    def get_max_input_tokens(self, model_name: str = None) -> int:
        """Get the max number of input tokens allowed for a model.

        Args:
            model_name: Model name to get max input token number for.

        Returns:
            int: The max number of input tokens allowed.
        """
        if self.max_input_tokens:
            return self.max_input_tokens
        if self.config.get("context_window_size"):
            self.max_input_tokens = int(self.config.get("context_window_size"))
            return self.max_input_tokens
        model = model_name or self.config.get("model")
        self.max_input_tokens = MODEL_CONTEXT_WINDOW_SIZE.get(model, DEFAULT_CONTEXT_WINDOW_SIZE)
        return self.max_input_tokens

    def response_to_text(self, response) -> str:
        """Return response object as text.

        Args:
            response: The response object from the Anthropic API.

        Returns:
            str: The response as text.
        """
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError, APIConnectionError),
        max_time=TEN_MINUTES,
        on_backoff=_backoff_handler,
    )
    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate text using the Anthropic API.

        Args:
            prompt: The prompt to use for the generation.
            as_object: Return response object from API if True, otherwise return text.

        Returns:
            str or object: Generated text as a string or response object from the API.
        """
        kwargs = {}
        system_blocks = self._system_blocks()
        if system_blocks:
            kwargs["system"] = system_blocks

        response = self.client.messages.create(
            model=self.config.get("model"),
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_config,
            **kwargs,
        )
        if as_object:
            return response
        return self.response_to_text(response)

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError, APIConnectionError),
        max_time=TEN_MINUTES,
        on_backoff=_backoff_handler,
    )
    def chat(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Chat using the Anthropic API.

        Args:
            prompt: The user prompt to chat with.
            as_object: Return response object from API if True, otherwise return text.

        Returns:
            str or object: Chat response as a string or response object from the API.
        """
        if not self.chat_session:
            self.chat_session = self.create_chat_session()
        response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return self.response_to_text(response)


# This module doubles as the base for the Vertex AI variant, which imports it
# even when the API-key configuration is absent; only register when usable.
if os.getenv("ANTHROPIC_API_KEY"):
    manager.LLMManager.register_provider(AnthropicProvider)
