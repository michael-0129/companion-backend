import httpx
from app.core.config import settings

class VllmProvider:
    """
    vLLM provider using OpenAI-compatible chat API.
    """
    def __init__(self):
        self.endpoint = settings.VLLM_API_URL.rstrip("/")
        self.model = settings.VLLM_MODEL
        self.max_tokens = settings.VLLM_MAX_TOKENS
        self.temperature = settings.VLLM_TEMPERATURE

    async def generate(self, messages, max_tokens=None, temperature=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.endpoint}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

def get_llm_provider():
    """
    Returns the vLLM provider for all LLM calls.
    """
    return VllmProvider()

# Documentation:
# To add a new provider, implement a new class inheriting from LLMProvider and add it to PROVIDER_MAP. 