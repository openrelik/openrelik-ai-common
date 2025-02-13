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

import os

PROVIDER_CONFIG = {
    "ollama": ["OLLAMA_SERVER_URL", "OLLAMA_DEFAULT_MODEL"],
    "vertexai": ["VERTEXAI_PROJECT_ID", "VERTEXAI_LOCATION", "VERTEXAI_DEFAULT_MODEL"],
    "googleai": ["GOOGLEAI_API_KEY", "GOOGLEAI_DEFAULT_MODEL"],
}


def get_provider_config(provider_name):
    """Get the configuration for a provider.

    Args:
        provider_name: The name of the provider.

    Returns:
        A dictionary of configuration values.
    """
    provider_config = {}
    for env_var in PROVIDER_CONFIG.get(provider_name):
        config_name = env_var.lower().split("_", 1)[1]
        config_value = os.getenv(env_var)
        provider_config[config_name] = config_value
    return provider_config
