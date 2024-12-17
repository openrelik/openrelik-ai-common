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

import vertexai
import google.generativeai as genai
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory

from . import interface, manager


class VertexAI(interface.LLMProvider):
    """Vertex AI LLM provider."""

    NAME = "vertexai"
    DISPLAY_NAME = "VertexAI"

    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vertexai.init(
            project=self.config.get("project_id"),
        )

        self.client = GenerativeModel(
            model_name=self.config.get("model"),
            generation_config=self.generation_config,
            system_instruction=self.config.get("system_instructions"),
            safety_settings=self.SAFETY_SETTINGS,
        )
        self.chat_session = self.create_chat_session()
    
    def create_chat_session(self):
        """Create chat session object."""
        return self.client.start_chat()

    @property
    def generation_config(self):
        return {
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p_sampling"),
            "top_k": self.config.get("top_k_sampling"),
        }

    def count_tokens(self, prompt: str) -> int:
        """
        Count the number of tokens in a prompt using the Vertex AI service.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            The number of tokens in the prompt.
        """
        return self.client.count_tokens(prompt).total_tokens
    
    def max_input_tokens(self, model_name: str) -> int:
        """
        Get the max number of input tokens allowed for a model.

        Args:
            model_name: Model name to get max input token number for.

        Returns:
            The max number of input tokens allowed.
        """
        if self.max_input_tokens:
            return self.max_input_tokens
        self.max_input_tokens = genai.get_model(
            f"models/{model_name}").input_token_limit
        return self.max_input_tokens

    def chat(
        self, prompt: str, file_content: str = None, as_object: bool = False
    ) -> Union[str, object]:
        """
        Generate text using the Vertex AI service.

        Args:
            prompt: The prompt to use for the generation.
            file_content: If file_content is provided and the overall prompt
                limit is more than maximum allowed input token count, then
                the file content will be split into chunks and iterative
                summary will be returned and used in the history session.
            as_object: return response object from API else text.

        Returns:
            The generated text as a string.
        """
        if file_content:
            response = self.do_chunked_prompt(prompt, file_content, self.generate)
        else:
            response = self.client.generate_content(prompt)
        if as_object:
            return response
        return response.text

    def chat(self, prompt: str, file_content: str = None, as_object: bool = False) -> Union[str, object]:
        """Chat using the Google AI service.

        Args:
            prompt: The user prompt to chat with.
            file_content: If file_content is provided and the overall prompt
                limit is more than maximum allowed input token count, then
                the file content will be split into chunks and iterative
                summary will be returned and used in the history session.
            as_object: return response object from API else text.

        Returns:
            The chat response.
        """
        if file_content:
            response = self.do_chunked_prompt(prompt, file_content, self.chat)
        else:
            response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return response.text


manager.LLMManager.register_provider(VertexAI)
