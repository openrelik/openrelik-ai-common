# tests/test_get_next_chunk.py

# --- stub out the LLM interface needed by TextFileChunker ---
class DummyLLM:
    def __init__(self):
        self.config = {"model": "dummy", "system_instructions": ""}
    def get_max_input_tokens(self, model_name):
        return 100  # or whatever token limit you want to test
    def count_tokens(self, text):
        return len(text)  # simplistic: 1 char = 1 token

# --- now import and test ---
from openrelik_ai_common.utils.chunker import TextFileChunker

def test_get_next_chunk_int_cast():
    content = "x" * 200
    dummy = DummyLLM()
    c = TextFileChunker(prompt="p", file_content=content, llm=dummy)
    # this used to raise a TypeError before the int() cast
    chunk, offset = c._get_next_chunk(prompt="p",
                                       prompt_chunk_wrapper="",
                                       offset=0.0)
    # verify the fix
    assert isinstance(offset, int)
    assert len(chunk) == offset
    assert offset <= dummy.get_max_input_tokens("") * 4