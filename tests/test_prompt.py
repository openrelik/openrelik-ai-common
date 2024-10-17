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

import unittest
from openrelik_ai_common.utils.prompt import prompt_from_template


class TestPrompt(unittest.TestCase):
    """Test the prompt_from_template function."""

    def test_prompt_from_template(self):
        """Test the prompt_from_template function."""
        template = "Hello, {name}! This is a {object}."
        kwargs = {"name": "world", "object": "test"}
        expected_prompt = "Hello, world! This is a test."
        self.assertEqual(prompt_from_template(template, kwargs), expected_prompt)

    def test_prompt_from_template_missing_key(self):
        """Test the prompt_from_template function with a missing key."""
        template = "Hello, {name}! This is a {object}."
        kwargs = {"name": "world"}
        with self.assertRaises(KeyError):
            prompt_from_template(template, kwargs)
