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

from . import interface, manager

from ollama import Client


class Ollama(interface.LLMProvider):
    """A LLM provider for the Ollama server."""

    NAME = "ollama"
    DISPLAY_NAME = "Ollama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(host=self.config.get("server_url"))

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

    def generate(self, prompt: str) -> str:
        """Generate text using the ollama server.

        Args:
            prompt: The prompt to use for the generation.

        Raises:
            ValueError: If the generation fails.

        Returns:
            The generated text as a string.
        """
        client = Client(host=self.config.get("server_url"))
        response = client.generate(
            prompt=prompt,
            model=self.config.get("model"),
            options={
                "temperature": self.config.get("temperature"),
                "num_predict": self.config.get("max_output_tokens"),
            },
        )
        return response.get("response")


manager.LLMManager.register_provider(Ollama)
