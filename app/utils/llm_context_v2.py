import logging
from typing import List, Dict, Optional, Any
from app.models import CodexEntry, ChatHistory
from app.core.config import settings
from app.utils.security import decrypt_content
from app.utils.token_counter import TokenCounter
from app.core.exceptions import InputTooLongError

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _has_tiktoken = True
except ImportError:
    _has_tiktoken = False
    logger.warning("tiktoken not installed; token counting will be approximate.")

class LlmContextManager:
    """
    Manages context for LLMs using OpenAI/vLLM chat format. Handles context assembly, token counting, and context window enforcement.
    Automatically trims memories and chat history if the context would exceed the model's max token count.
    """
    def __init__(self, model_name: str = settings.VLLM_MODEL, max_input_tokens: int = settings.VLLM_MAX_INPUT_TOKENS, max_output_tokens: int = settings.VLLM_MAX_OUTPUT_TOKENS):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.system_prompt: Optional[str] = None
        self.memories: List[str] = []
        self.chat_history: List[str] = []
        self.query: Optional[str] = None
        self.token_counter = TokenCounter(model_name)
        self.warning = None

    def set_system_prompt(self, prompt: str):
        """Sets the system prompt."""
        self.system_prompt = prompt

    def add_query(self, query: str):
        """Adds the user's query to the context."""
        self.query = query

    def add_memory(self, memory_content: str):
        """Adds a single memory to the context."""
        self.memories.append(memory_content)

    def add_memories(self, memories: List[CodexEntry]):
        for memory in memories:
            try:
                decrypted_content = decrypt_content(memory.encrypted_content)
                memory_content = f"Memory from {memory.event_date.strftime('%Y-%m-%d')}: {decrypted_content}"
                self.add_memory(memory_content)
            except Exception as e:
                logger.error(f"Failed to decrypt or add memory {memory.id}: {e}")

    def add_chat_history(self, chat_history: List[ChatHistory]):
        for entry in reversed(chat_history):
            chat_pair = f"User: {entry.user_query}\nAI: {entry.companion_response}"
            self.chat_history.insert(0, chat_pair)

    def _get_token_count(self, messages: List[Dict[str, str]]) -> int:
        """
        Returns the token count for the given messages using the modular TokenCounter utility.
        """
        try:
            return self.token_counter.count_messages(messages)
        except Exception as e:
            logger.error(f"[LLM_CONTEXT] Token counting failed for model '{self.model_name}': {e}")
            # Fallback: 1 token ≈ 0.75 words
            return sum(len(m['content'].split()) for m in messages) // 0.75

    def get_classify_messages(self) -> List[Dict[str, str]]:
        # Assemble system prompt and user query only; do not summarize or truncate (already checked at API entry)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_msg = {"role": "user", "content": self.query}
        temp_messages = messages + [user_msg]
        token_count = self._get_token_count(temp_messages)
        if token_count > self.max_input_tokens:
            # This should never happen due to early API check, but raise for protocol safety
            self.warning = f"User query and system prompt exceed the input token limit ({self.max_input_tokens})."
            raise InputTooLongError(self.warning)
        return temp_messages

    def get_answer_messages(self) -> List[Dict[str, str]]:
        # Assemble context for answer; trim chat history and memories if needed, but do not summarize
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        memories = list(self.memories)
        chat_history = list(self.chat_history)
        user_query = self.query
        while True:
            user_content = ""
            if memories:
                user_content += "[Relevant Memories]:\n" + "\n".join(memories) + "\n\n"
            if chat_history:
                user_content += "[Recent Chat History]:\n" + "\n".join(chat_history) + "\n\n"
            if user_query:
                user_content += user_query
            temp_messages = messages + ([{"role": "user", "content": user_content.strip()}] if user_content.strip() else [])
            token_count = self._get_token_count(temp_messages)
            if token_count <= self.max_input_tokens:
                if user_content.strip():
                    messages.append({"role": "user", "content": user_content.strip()})
                break
            # Trim chat history first, then memories
            if chat_history:
                logger.info(f"[CONTEXT] Trimming chat history to fit input token limit ({token_count} > {self.max_input_tokens})")
                chat_history.pop(0)
            elif memories:
                logger.info(f"[CONTEXT] Trimming memories to fit input token limit ({token_count} > {self.max_input_tokens})")
                memories.pop(0)
            else:
                logger.critical(f"[CONTEXT] Unable to fit context within input token limit ({token_count} > {self.max_input_tokens}) even after trimming.")
                self.warning = f"Context cannot be reduced below input token limit ({self.max_input_tokens})!"
                raise InputTooLongError(self.warning)
        return messages

    def get_warning(self):
        return self.warning 