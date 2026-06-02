# Copyright 2026 Google LLC
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

"""Anthropic Claude on Vertex AI LLM provider.

Serves Claude models from Vertex AI Model Garden using Application Default
Credentials (e.g. GKE Workload Identity or a service account), so no API key
needs to be provisioned or stored.
"""

from anthropic import AnthropicVertex

from . import manager
from .anthropic import AnthropicProvider


class AnthropicVertexProvider(AnthropicProvider):
    """Anthropic Claude provider backed by Vertex AI (ADC auth, no API key)."""

    NAME = "anthropicvertex"
    DISPLAY_NAME = "Anthropic Claude (Vertex AI)"

    def _create_client(self):
        """Create a Vertex AI client authenticated via ADC."""
        return AnthropicVertex(
            project_id=self.config.get("project_id"),
            region=self.config.get("region"),
        )


manager.LLMManager.register_provider(AnthropicVertexProvider)
