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
        response = self.client.chat(
            messages=self.history,
            model=self.model,
        )
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
        self.chat_session = ChatSession(
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

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate text using the ollama server.

        Args:
            prompt: The prompt to use for the generation.

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
                "num_predict": self.config.get("max_output_tokens"),
            },
        )
        if as_object:
            return response
        return response.get("response")

    def chat(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Chat using the ollama server.

        Args:
            prompt: The user prompt to chat with.

        Returns:
            The chat response.
        """
        response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return response.get("message", {}).get("content")


manager.LLMManager.register_provider(Ollama)
