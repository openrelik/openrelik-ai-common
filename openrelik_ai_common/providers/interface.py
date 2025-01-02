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
from typing import Union

from ..utils.chunker import TextFileChunker
from .config import get_provider_config

# Default values
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P_SAMPLING = 0.1
DEFAULT_TOP_K_SAMPLING = 1


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

    def get_max_input_tokens(self, model_name: str):
        """Get the max number of input tokens allowed for a model.

        Args:
            model_name: Model name to get max input token number for.

        Returns:
            The max number of input tokens allowed.
        """
        raise NotImplementedError()

    def response_to_text(self, response):
        """Return response object as text.

        Args:
            response: The response object from the LLM provider.

        Returns:
            The response as text.
        """
        return NotImplementedError()

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate a response from the LLM provider.

        Args:
            prompt: The prompt to generate a response for.
            as_object: return response object from API else text.

        Returns:
            The generated response.
        """
        raise NotImplementedError()

    def generate_file_analysis(self, prompt: str, file_content: str = None) -> str:
        """Analyze file content using the LLM provider.

        Args:
            prompt: The prompt to analyze the file content with.
            file_content: The content of the file to analyze.

        Returns:
            The generated file analysis.
        """
        chunker = TextFileChunker(
            prompt=prompt,
            file_content=file_content,
            llm=self,
        )
        response = chunker.process_file_content()
        return self.response_to_text(response)

    def chat(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Chat using the LLM provider.

        Args:
            prompt: The user prompt to chat with.
            as_object: return response object from API else text.

        Returns:
            The chat response.
        """
        raise NotImplementedError()
