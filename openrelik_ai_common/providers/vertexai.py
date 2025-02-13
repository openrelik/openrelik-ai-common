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
"""Vertex AI LLM provider."""
from typing import Union

from google import genai
from google.genai import types

from . import interface, manager


class VertexAI(interface.LLMProvider):
    """Vertex AI LLM provider."""

    NAME = "vertexai"
    DISPLAY_NAME = "VertexAI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.client = genai.Client(
            vertexai=True,
            project=self.config.get("project_id"),
            location=self.config.get("location")
        )

        self.chat_session = self.create_chat_session()

    def create_chat_session(self):
        """Create chat session object."""
        return self.client.chats.create(
            model=self.config.get("model"),
            config=types.GenerateContentConfig(
                system_instruction=self.config.get("system_instructions"),
                temperature=self.config.get("temperature"),
                top_p=self.config.get("top_p_sampling"),
                top_k=self.config.get("top_k_sampling")
            ),
        )

    def count_tokens(self, prompt: str) -> int:
        """
        Count the number of tokens in a prompt using the Vertex AI service.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            The number of tokens in the prompt.
        """
        return self.client.models.count_tokens(
            model=self.config.get("model"),
            contents=prompt).total_tokens


    def get_max_input_tokens(self, model_name: str) -> int:
        """
        Get the max number of input tokens allowed for a model.

        Args:
            model_name: Model name to get max input token number for.

        Returns:
            The max number of input tokens allowed.
        """
        if self.max_input_tokens:
            return self.max_input_tokens

        # Conservative estimate for Gemini models. No support for model lookup and token
        # limit retrieval at the moment.
        self.max_input_tokens = 128000

        return self.max_input_tokens

    def response_to_text(self, response):
        """Return response object as text.

        Args:
            response:
                The response object from the Google AI service.

        Returns:
            str:
                The response as text.
        """
        return response.text

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """
        Generate text using the Vertex AI service.

        Args:
            prompt: The prompt to use for the generation.
            as_object: return response object from API else text.

        Returns:
            The generated text as a string.
        """
        response = self.client.models.generate_content(
            model=self.config.get("model"),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.config.get("system_instructions"),
                temperature=self.config.get("temperature"),
                top_p=self.config.get("top_p_sampling"),
                top_k=self.config.get("top_k_sampling")
            ),
        )

        if as_object:
            return response
        return self.response_to_text(response)

    def chat(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Chat using the Google AI service.

        Args:
            prompt: The user prompt to chat with.
            as_object: return response object from API else text.
            chat_session: Optional chat session object.

        Returns:
            The chat response.
        """
        response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return self.response_to_text(response)


manager.LLMManager.register_provider(VertexAI)
