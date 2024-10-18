# openrelik-ai-common
Common utilities for OpenRelik AI functionality.

```python
# LLM providers are configured via environment variables
# export OLLAMA_SERVER_URL=http://localhost:11434
# export OLLAMA_DEFAULT_MODEL=gemma2:9b

from openrelik_ai_common.providers import manager

llm = manager.LLMManager().get_provider("ollama")()
response = llm.generate(prompt="Hello")
print(response)
```

##### Obligatory Fine Print
This is not an official Google product (experimental or otherwise), it is just code that happens to be owned by Google.
