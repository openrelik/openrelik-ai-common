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
"""Google AI LLM provider."""
import google.generativeai as genai

from . import interface, manager


class GoogleAI(interface.LLMProvider):
    """Google AI LLM provider."""

    NAME = "googleai"
    DISPLAY_NAME = "Google AI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.config.get("api_key"))
        self.client = genai.GenerativeModel(self.config.get("model"))

    def count_tokens(self, prompt: str):
        """
        Count the number of tokens in a prompt using the Google AI service.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            The number of tokens in the prompt.
        """
        return self.client.count_tokens(prompt).total_tokens

    def generate(self, prompt: str) -> str:
        """
        Generate text using the Google AI service.

        Args:
            prompt: The prompt to use for the generation.

        Returns:
            The generated text as a string.
        """
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.config.get("max_output_tokens"),
                temperature=self.config.get("temperature"),
            ),
        )
        return response.text


manager.LLMManager.register_provider(GoogleAI)
