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

from .config import get_provider_config

# Default values
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P_SAMPLING = 0.1
DEFAULT_TOP_K_SAMPLING = 1
DEFAULT_MAX_OUTPUT_TOKENS = 2048


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
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        """Initialize the LLM provider.

        Args:
            model_name: The name of the model to use.
            system_instructions: The system instruction to use for the response.
            temperature: The temperature to use for the response.
            top_p_sampling: The top_p sampling to use for the response.
            top_k_sampling: The top_k sampling to use for the response.
            max_output_tokens: The maximum number of output tokens to generate.

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
        config["max_output_tokens"] = max_output_tokens

        # Load the LLM provider config from environment variables.
        config_from_environment = get_provider_config(self.NAME)
        if not config_from_environment:
            raise Exception(f"{self.NAME} config not found")
        config.update(config_from_environment)

        if not model_name:
            config["model"] = config.get("default_model")

        # Expose the config as an attribute.
        self.config = config

        # Create chat session.
        self.chat_session = None

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
                "max_output_tokens": self.config.get("max_output_tokens"),
            },
        }

    @property
    def generation_config(self):
        """Get the generation config for the LLM provider.

        Returns:
            A dictionary representation of the generation config.
        """
        return NotImplementedError()

    def count_tokens(self, prompt: str):
        """
        Count the number of tokens in a prompt.

        Args:
            prompt: The prompt to count the tokens for.

        Returns:
            The number of tokens in the prompt.
        """
        raise NotImplementedError()

    def generate(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Generate a response from the LLM provider.

        Args:
            prompt: The prompt to generate a response for.

        Returns:
            The generated response.
        """
        raise NotImplementedError()

    def chat(self, prompt: str, as_object: bool = False) -> Union[str, object]:
        """Chat using the LLM provider.

        Args:
            prompt: The user prompt to chat with.

        Returns:
            The chat response.
        """
        raise NotImplementedError()
