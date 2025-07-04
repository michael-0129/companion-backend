import logging
import requests
import re

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False
    logger.warning("tiktoken not installed; OpenAI token counting will be unavailable.")

try:
    from transformers import AutoTokenizer
    _has_transformers = True
except ImportError:
    _has_transformers = False
    logger.warning("transformers not installed; HF model token counting will be unavailable.")

# Sanitize function to remove control characters except \n, \r, \t
def sanitize_for_json(text):
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

MODEL_REGISTRY = {
    "llama": {
        "tokenizer": "vllm_api",
        "vllm_tokenizer_url": "http://localhost:8080",
        "max_tokens": 8192,
    },
    "mixtral": {
        "tokenizer": "vllm_api",
        "vllm_tokenizer_url": "http://localhost:8080",
        "max_tokens": 32768,
    },
    "gpt": {
        "tokenizer": "tiktoken",
        "max_tokens": 4096,
    },
    # Add more as needed
}

def get_model_family(model_name: str) -> str:
    model_name = model_name.lower()
    for key in MODEL_REGISTRY:
        if key in model_name:
            return key
    return "gpt"  # Default fallback

class TokenCounter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.family = get_model_family(model_name)
        self.config = MODEL_REGISTRY[self.family]
        self.tokenizer_type = self.config["tokenizer"]
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.vllm_tokenizer_url = self.config.get("vllm_tokenizer_url")
        # Optionally, add local fallback here if needed
        if self.tokenizer_type == "tiktoken":
            try:
                import tiktoken
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except Exception as e:
                logger.warning(f"[TokenCounter] Unknown model '{model_name}' for tiktoken; using 'cl100k_base'. Exception: {e}")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None

    def count(self, text: str) -> int:
        if self.tokenizer_type == "vllm_api":
            return self._count_vllm_api(text)
        elif self.tokenizer_type == "tiktoken":
            return len(self.tokenizer.encode(text))
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def count_messages(self, messages) -> int:
        if self.tokenizer_type == "vllm_api":
            # Concatenate all content for a rough estimate
            all_text = "\n".join(m.get("content", "") for m in messages)
            return self._count_vllm_api(all_text)
        elif self.tokenizer_type == "tiktoken":
            num_tokens = 0
            for m in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in m.items():
                    num_tokens += len(self.tokenizer.encode(value))
            return num_tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def _count_vllm_api(self, text: str) -> int:
        if not self.vllm_tokenizer_url:
            logger.error("[TokenCounter] vLLM tokenizer URL not configured.")
            raise RuntimeError("vLLM tokenizer URL not configured.")
        try:
            sanitized_text = sanitize_for_json(text)
            response = requests.post(
                self.vllm_tokenizer_url,
                json={"prompt": sanitized_text, "model": self.model_name},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            # Accept both 'tokens' and 'token_ids'
            tokens = data.get("tokens")
            if tokens is not None:
                print(f"[LENGTH] {len(tokens)}")
                return len(tokens)
            token_ids = data.get("token_ids")
            if token_ids is not None:
                print(f"[LENGTH] {len(token_ids)}")
                return len(token_ids)
            logger.error(f"[TokenCounter] Unexpected vLLM tokenizer API response: {data}")
            raise RuntimeError("Unexpected vLLM tokenizer API response.")
        except Exception as e:
            logger.error(f"[TokenCounter] vLLM tokenizer API call failed: {e}")
            return max(1, len(text.split()) // 0.75) 