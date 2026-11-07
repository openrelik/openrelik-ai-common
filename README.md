> [!IMPORTANT]
> **This repository has moved to [openrelik/openrelik](https://github.com/openrelik/openrelik),**
> the OpenRelik monorepo, where it lives in [`libs/ai-common/`](https://github.com/openrelik/openrelik/tree/main/libs/ai-common).
> This repository is archived and read-only.
>
> The full history moved with it, and its tags are there as `ai-common-<tag>`. Open issues were moved too. Please open new issues and pull requests in
> [openrelik/openrelik](https://github.com/openrelik/openrelik).

### Common utilities for OpenRelik AI functionality.

```python
# LLM providers are configured via environment variables
# export OLLAMA_SERVER_URL=http://localhost:11434
# export OLLAMA_DEFAULT_MODEL=gemma2:9b

from openrelik_ai_common.providers import manager

provider = manager.LLMManager().get_provider("ollama")
llm = provider(model_name="gemma2:9b", system_instructions="Your name is Foobar.")

# Single text generation
response = llm.generate(prompt="Hello, what is your name?")
print(response)

# Multiturn chat session
response = llm.chat("What is your name?")
print(response)
response = llm.chat("My name is John Doe")
print(response)
response = llm.chat("What is my name?")
print(response)

# Chat on the command line
while True:
    message = input("Message: ")
    response = llm.chat(message)
    print(response)
```

##### Obligatory Fine Print
This is not an official Google product (experimental or otherwise), it is just code that happens to be owned by Google.
