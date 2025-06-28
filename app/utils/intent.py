"""
Intent classification utility (LLM-based + rules-based for greetings).
Classifies user messages as 'greeting', 'inner_state', 'command', 'factual', or 'multi_intent'.
"""
from typing import Optional
from app.core.config import settings
import logging
import re
from app.utils.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

INTENT_TYPES = [
    "greeting",     # Explicit greeting (never triggers posture)
    "inner_state",  # Emotional, posture-relevant
    "command",      # Direct command or action request
    "factual",      # Factual statement or information
    "multi_intent", # Contains both emotional and command/factual
]

# Simple rules-based greeting detection (expand as needed)
GREETING_PATTERNS = [
    r"^\s*hi\b",
    r"^\s*hello\b",
    r"^\s*hey\b",
    r"^\s*good (morning|afternoon|evening|day)\b",
    r"^\s*howdy\b",
    r"^\s*greetings\b",
    r"^\s*yo\b",
    r"^\s*sup\b",
    r"^\s*what's up\b",
]

def is_greeting(message: str) -> bool:
    msg = message.strip().lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, msg):
            return True
    return False

INTENT_PROMPT = (
    "You are a mythic, protocol-driven AI Companion. "
    "Classify the following user message as one of: 'greeting', 'inner_state', 'command', 'factual', or 'multi_intent'. "
    "- 'greeting': The message is a greeting (e.g., 'Hi', 'Hello', 'Good morning'). Never treat a greeting as posture.\n"
    "- 'inner_state': The message expresses an emotion, uncertainty, or inner posture.\n"
    "- 'command': The message is a direct command, request for action, or protocol instruction.\n"
    "- 'factual': The message is a factual statement or information, not a command or emotion.\n"
    "- 'multi_intent': The message contains both an inner state and a command/action.\n"
    "Respond ONLY with the label. Do not explain or add extra text.\n"
    "Message: \"{message}\"\nIntent:"
)

async def classify_intent_from_message_llm(message: str) -> Optional[str]:
    """
    Classifies the user's message intent. Returns one of INTENT_TYPES, or None if not detected.
    Rules-based for greetings, LLM for others.
    """
    if is_greeting(message):
        logger.info(f"[INTENT] Detected greeting: '{message}'")
        return "greeting"
    prompt = INTENT_PROMPT.format(message=message)
    try:
        llm = get_llm_provider()
        response = await llm.generate(prompt=prompt, max_tokens=settings.VLLM_MAX_TOKENS, temperature=settings.VLLM_TEMPERATURE)
        content = response.strip().lower()
        for intent in INTENT_TYPES:
            if intent in content:
                return intent
        return content if content else None
    except Exception as e:
        logger.error(f"Intent LLM classification failed: {e}", exc_info=True)
        return None 