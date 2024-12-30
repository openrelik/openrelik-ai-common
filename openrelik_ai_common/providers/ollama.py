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
"""LLM provider for the ollama server."""

from typing import Union

from ollama import Client

from . import interface, manager

DEFAULT_MAX_OUTPUT_TOKENS = 2048
MODEL_CONTEXT_WINDOW_SIZE = {
    "gemma2:9b": 8192,
}


class ChatSession:
    """A chat session with the ollama server."""

    def __init__(self, client: Client, model_name: str, system: str = None):
        self.client = client
        self.model = model_name
        self.history = []

        # Set the system instructions. Ollama doesn't support this for chats, only for
        # generating text. We work around this by setting an explicit pre-prompt in the
        # chat history.
        if system:
            self.history.append({"role": "system", "content": system})

    def send_message(self, prompt: str):
        """Send a message to the chat session.

        Args:
            message: The message to send.

        Returns:
            The response from the chat session.
        """
        message = {"role": "user", "content": prompt}
        self.history.append(message)
        response = self.client.chat(messages=self.history, model=self.model)
        self.history.append(response.get("message", {}))
        return response


class Ollama(interface.LLMProvider):
    """A LLM provider for the Ollama server."""

    NAME = "ollama"
    DISPLAY_NAME = "Ollama"

    def __init__(self, **kwargs):
        """Initialize the Ollama provider.

        Attributes:
            client: The Ollama client.
            chat_session: A chat session.
        """
        super().__init__(**kwargs)
        self.client = Client(host=self.config.get("server_url"))
        self.chat_session = self.create_chat_session()

    def create_chat_session(self):
        """Create a chat session with the Ollama server."""
        return ChatSession(
            client=self.client,
            model_name=self.config.get("model"),
            system=self.config.get("system_instructions"),
        )

    def count_tokens(self, prompt: str):
        """
        Count the number of tokens in a prompt. This is an estimate.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            The number of tokens in the prompt.
        """
        # Rough estimate: ~4chars UTF8, 1bytes per char.
        return len(prompt) / 4

    def get_max_input_tokens(self, model_name: str = None):
        """
        Get the max number of input tokens allowed for a model.

        Args:
            model_name: Model name to get max input token number for.

        Returns:
            The max number of input tokens allowed.
        """
        if self.max_input_tokens:
            return self.max_input_tokens

        if model_name in MODEL_CONTEXT_WINDOW_SIZE:
            print(
                "using context window size for model",
                model_name,
                MODEL_CONTEXT_WINDOW_SIZE[model_name],
            )
            self.max_input_tokens = MODEL_CONTEXT_WINDOW_SIZE[model_name]
        else:
            self.max_input_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        return DEFAULT_MAX_OUTPUT_TOKENS

    def response_to_text(self, response):
        """
        Extracts text content from a response object.

        Args:
            response: The response object, which can have different structures.

        Returns:
            The extracted text content, or None if no text could be found.
        """
        if isinstance(response, dict):
            message_content = response.get("message", {}).get("content")
            if message_content:
                return message_content

            response_value = response.get("response")
            if response_value:
                return response_value

        elif isinstance(response, str):  # if response is already a string, return it
            return response

        return None  # Return None if no suitable structure is found

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate text using the ollama server.

        Args:
            prompt: The prompt to use for the generation.
            as_object: return response object from API else text.

        Raises:
            ValueError: If the generation fails.

        Returns:
            The generated text as a string.
        """

        response = self.client.generate(
            prompt=prompt,
            model=self.config.get("model"),
            system=self.config.get("system_instructions"),
            options={
                "temperature": self.config.get("temperature"),
                "num_predict": DEFAULT_MAX_OUTPUT_TOKENS,
            },
        )
        if as_object:
            return response
        return self.response_to_text(response)

    def chat(
        self, prompt: str, as_object: bool = False, chat_session: object = None
    ) -> Union[str, object]:
        """Chat using the ollama server.

        Args:
            prompt: The user prompt to chat with.
            as_object: return response object from API else text.

        Returns:
            The chat response.
        """
        if not chat_session:
            chat_session = self.chat_session
        response = chat_session.send_message(prompt)
        if as_object:
            return response
        return self.response_to_text(response)


manager.LLMManager.register_provider(Ollama)
