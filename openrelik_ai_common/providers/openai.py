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

"""OpenAI LLM provider."""

import logging
from typing import Union

import backoff
import tiktoken
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

from . import interface, manager

CALL_LIMIT = 60  # Number of calls to allow within a period
ONE_MINUTE = 60  # One minute in seconds
TEN_MINUTES = 10 * ONE_MINUTE

# Context window sizes for common OpenAI models.
# Note: Can be overridden by OPENAI_CONTEXT_WINDOW_SIZE.
MODEL_CONTEXT_WINDOW_SIZE = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}
DEFAULT_MAX_INPUT_TOKENS = 8192

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _backoff_handler(details) -> None:
    """Backoff handler for OpenAI API calls."""
    logger.info("Backing off %s seconds after %s tries", details["wait"], details["tries"])


class ChatSession:
    """A chat session with the OpenAI API."""

    def __init__(self, client: OpenAI, model_name: str, system: str = None):
        self.client = client
        self.model = model_name
        self.history = []

        if system:
            self.history.append({"role": "developer", "content": system})

    def send_message(self, prompt: str):
        """Send a message to the chat session.

        Args:
            prompt: The user message to send.

        Returns:
            The response from the OpenAI API.
        """
        self.history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
        )
        assistant_message = response.choices[0].message
        self.history.append({"role": "assistant", "content": assistant_message.content})
        return response


class OpenAIProvider(interface.LLMProvider):
    """OpenAI LLM provider."""

    NAME = "openai"
    DISPLAY_NAME = "OpenAI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("api_base_url"),
        )
        self.chat_session = self.create_chat_session()

    @property
    def generation_config(self) -> dict:
        """Get the generation config for the OpenAI provider."""
        return {
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p_sampling"),
        }

    def create_chat_session(self) -> ChatSession:
        """Create a chat session with the OpenAI API."""
        return ChatSession(
            client=self.client,
            model_name=self.config.get("model"),
            system=self.config.get("system_instructions"),
        )

    def count_tokens(self, prompt: str) -> int:
        """Count the number of tokens in a prompt.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            int: The number of tokens in the prompt.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.config.get("model"))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(prompt))

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
        self.max_input_tokens = MODEL_CONTEXT_WINDOW_SIZE.get(model, DEFAULT_MAX_INPUT_TOKENS)
        return self.max_input_tokens

    def response_to_text(self, response) -> str:
        """Return response object as text.

        Args:
            response: The response object from the OpenAI API.

        Returns:
            str: The response as text.
        """
        return response.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError, APIConnectionError),
        max_time=TEN_MINUTES,
        on_backoff=_backoff_handler,
    )
    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate text using the OpenAI API.

        Args:
            prompt: The prompt to use for the generation.
            as_object: return response object from API else text.

        Returns:
            str or object: Generated text as a string or response object from the API.
        """
        messages = []
        if self.config.get("system_instructions"):
            messages.append(
                {"role": "developer", "content": self.config.get("system_instructions")}
            )
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.get("model"),
            messages=messages,
            **self.generation_config,
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
        """Chat using the OpenAI API.

        Args:
            prompt: The user prompt to chat with.
            as_object: return response object from API else text.

        Returns:
            str or object: Chat response as a string or response object from the API.
        """
        response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return self.response_to_text(response)


manager.LLMManager.register_provider(OpenAIProvider)
