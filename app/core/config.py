from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Configuration for the AI Companion Backend.
    Only vLLM/OpenAI-compatible LLM and local embedding settings are included.
    All legacy/unused fields have been removed for clarity and maintainability.
    """
    PROJECT_NAME: str = "AI Companion Backend"
    API_V1_STR: str = "/api/v1"

    # LLM Provider Configuration (vLLM/OpenAI-compatible only)
    LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'vllm')  # e.g., 'vllm', 'openai'
    VLLM_API_URL: str = os.getenv('VLLM_API_URL', 'http://localhost:8080')
    VLLM_MODEL: str = os.getenv('VLLM_MODEL', 'llama3.1-8b')
    VLLM_MAX_TOKENS: int = int(os.getenv('VLLM_MAX_TOKENS', 4096))
    VLLM_TEMPERATURE: float = float(os.getenv('VLLM_TEMPERATURE', 0.7))

    # New: Separate input/output token limits
    VLLM_MAX_INPUT_TOKENS: int = int(os.getenv('VLLM_MAX_INPUT_TOKENS', 3072))
    @property
    def VLLM_MAX_OUTPUT_TOKENS(self) -> int:
        return self.VLLM_MAX_TOKENS - self.VLLM_MAX_INPUT_TOKENS

    # Context Management
    CONTEXT_CHAT_HISTORY_LIMIT: int = 10  # Max chat history turns for context
    MAX_RECENT_CHAT_HISTORY: int = 10     # Max recent chat turns for context
    MAX_SEMANTIC_SEARCH_RESULTS: int = 5  # Max RAG search results
    MAX_DOC_TEXT_FOR_LLM_EXTRACTION_CHARS: int = 75000  # Max doc chars for LLM extraction
    DEFAULT_MAX_RESPONSE_TOKENS: int = 1500  # Max tokens for LLM response
    DEFAULT_SYSTEM_PROMPT_TOKEN_OVERHEAD: int = 100  # Overhead for system prompt

    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://postgres:root@127.0.0.1:5432/companion')

    # Security
    ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY')  # Fernet encryption key

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')  # e.g., DEBUG, INFO, WARNING, ERROR
    DEBUG_MODE: bool = bool(os.getenv('DEBUG_MODE', False))

    # Error/Protocol Responses
    DEFAULT_ERROR_RESPONSE: str = "I encountered an internal error. Please try again later."
    PRE_PROCESSING_ERROR_RESPONSE: str = "I encountered an issue before fully processing your request. This has been logged."
    SILENCE_MODE_RESPONSE: str = ""  # Default silent response is empty

    # Canonical Timezone (CET)
    TIMEZONE: str = "Europe/London"

    # Embedding Model (local, sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'intfloat/e5-small-v2')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'  # Ignore extra fields from .env

settings = Settings()

# Canonical CET timezone
TIMEZONE = getattr(settings, 'TIMEZONE', 'Europe/Berlin') 