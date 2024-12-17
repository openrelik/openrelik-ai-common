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

import logging
from typing import Union

import backoff
from google.api_core import exceptions
import google.generativeai as genai
from google.generativeai.types import generation_types
from google.generativeai.types import HarmBlockThreshold
from google.generativeai.types import HarmCategory
import ratelimit

from . import interface
from . import manager


CALL_LIMIT = 20   # Number of calls to allow within a period
ONE_MINUTE = 60   # One minute in seconds
TEN_MINUTE = 10 * ONE_MINUTE


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _backoff_hdlr(details) -> None:
    """Backoff handler for Google Generative AI calls."""
    logger.info("Backing off %s seconds after %s tries",
                details["wait"],
                details["tries"])


class GoogleAI(interface.LLMProvider):
    """Google AI LLM provider."""

    NAME = "googleai"
    DISPLAY_NAME = "Google AI"

    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
            HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
            HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        HarmCategory.HARM_CATEGORY_HARASSMENT: (
            HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
            HarmBlockThreshold.BLOCK_ONLY_HIGH
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.config.get("api_key"))
        self.client = genai.GenerativeModel(
            model_name=self.config.get("model"),
            generation_config=self.generation_config,
            system_instruction=self.config.get("system_instructions"),
            safety_settings=self.SAFETY_SETTINGS,
        )
        self.chat_session = self.create_chat_session()

    def create_chat_session(self) -> genai.ChatSession:
        """Create a chat session with the Google AI service."""
        return self.client.start_chat()

    @property
    def generation_config(self) -> dict[str, float]:
        """Get the generation config for the Google AI service."""
        return {
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p_sampling"),
            "top_k": self.config.get("top_k_sampling"),
        }

    def count_tokens(self, prompt: str) -> int:
        """Count the number of tokens in a prompt using the Google AI service.

        Args:
            prompt:
                The prompt to count the tokens for.

        Returns:
            int:
                The number of tokens in the prompt.
        """
        return self.client.count_tokens(prompt).total_tokens

    def max_input_tokens(self, model_name: str) -> int:
        """Get the max number of input tokens allowed for a model.

        Args:
            model_name:
                Model name to get max input token number for.

        Returns:
            int:
                The max number of input tokens allowed.
        """
        if self.max_input_tokens:
            return self.max_input_tokens
        self.max_input_tokens = genai.get_model(
            f"models/{model_name}"
        ).input_token_limit
        return self.max_input_tokens

    # Retry with exponential backoff strategy when exceptions occur
    @backoff.on_exception(
        backoff.expo,
        (
            exceptions.ResourceExhausted,
            exceptions.ServiceUnavailable,
            exceptions.GoogleAPIError,
            exceptions.InternalServerError,
            exceptions.Cancelled,
            ratelimit.RateLimitException,
        ),  # Exceptions to retry on
        max_time=TEN_MINUTE,
        on_backoff=_backoff_hdlr,  # Function to call when retrying
    )
    # Limit the number of calls to the model per minute
    @ratelimit.limits(calls=CALL_LIMIT, period=ONE_MINUTE)
    def generate(
        self, prompt: str, file_content: str = None, as_object: bool = False
    ) -> Union[str, generation_types.GenerateContentResponse]:
        """Generate text using the Google AI service.

        Args:
            prompt:
                The prompt to use for the generation.
            file_content:
                If file_content is provided and the overall prompt limit
                is more than maximum allowed input token count, then the file
                content will be split into chunks and iterative summary will be
                returned and used in the history session.
            as_object:
                return response object from API else text.

        Returns:
            str or object:
                The generated text as a string or the response object from the
                API.
        """
        if file_content:
            response = self.do_chunked_prompt(
                prompt, file_content, self.generate
            )
        else:
            response = self.client.generate_content(prompt)
        if as_object:
            return response
        return ",".join(
            [part.text for part in response.candidates[0].content.parts]
        )

    # Retry with exponential backoff strategy when exceptions occur
    @backoff.on_exception(
        backoff.expo,
        (
            exceptions.ResourceExhausted,
            exceptions.ServiceUnavailable,
            exceptions.GoogleAPIError,
            exceptions.InternalServerError,
            exceptions.Cancelled,
            ratelimit.RateLimitException,
        ),  # Exceptions to retry on
        max_time=TEN_MINUTE,
        on_backoff=_backoff_hdlr,  # Function to call when retrying
    )
    # Limit the number of calls to the model per minute
    @ratelimit.limits(calls=CALL_LIMIT, period=ONE_MINUTE)
    def chat(
        self, prompt: str, file_content: str = None, as_object: bool = False
    ) -> Union[str, generation_types.GenerateContentResponse]:
        """Chat using the Google AI service.

        Args:
            prompt:
                The user prompt to chat with.
            file_content:
                If file_content is provided and the overall prompt limit
                is more than maximum allowed input token count, then the file
                content will be split into chunks and iterative summary will be
                returned and used in the history session.
            as_object:
                return response object from API else text.

        Returns:
            str or object:
                The chat response as a string or the response object from the
                API.
        """
        self._ensure_nonempty_chat_parts()
        if file_content:
            response = self.do_chunked_prompt(prompt, file_content, self.chat)
        else:
            response = self.chat_session.send_message(prompt)
        if as_object:
            return response
        return ",".join(
            [part.text for part in response.candidates[0].content.parts]
        )

    def _ensure_nonempty_chat_parts(self):
        """Ensures that chat history parts are not empty for proto validation.

        Since this is a multi-turn conversation, the history is sent with each
        new request, the model's reply sent in the history with the next message
        can't be empty else the proto validators will complain behind the scene.
        However in some cases the model sends an empty content, we patch it and
        replace it with an ack message to avoid erroring out when re-sending the
        empty content in history with the next message.
        """
        history = self.chat_session.history
        history_patched = []
        for content in history:
            if not content.parts:
                content.parts = [genai.types.content_types.to_part("ack")]
            history_patched.append(content)
        self.chat_session.history = history_patched


manager.LLMManager.register_provider(GoogleAI)
