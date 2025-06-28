"""
Posture inference utility (LLM-based).
Provides a function to infer the user's conversational posture (e.g., clarity, confusion, curiosity) from a message and context using an LLM.
"""
from typing import Optional
from app.utils.llm_provider import get_llm_provider
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Canonical posture types
POSTURE_TYPES = [
    "clarity",
    "confusion",
    "curiosity",
    "hesitation",
    "confidence",
    "reflection",
    "uncertainty",
]

POSTURE_PROMPT = (
    "You are a mythic, protocol-driven AI Companion. "
    "Given the recent conversation context and the user's latest message, infer the user's true underlying posture of mind—not just the surface tone of the last message. "
    "Possible postures include: clarity, confusion, curiosity, hesitation, confidence, reflection, uncertainty. "
    "Consider the user's emotional trajectory, shifts, and the deeper state they are expressing, even if not explicit. "
    "If the posture is ambiguous, choose the closest match. Do not explain or add extra text.\n"
    "Recent context: {context}\n"
    "Latest message: \"{message}\"\nPosture:"
)

def _get_recent_context_snippet(context: Optional[str]) -> str:
    if not context:
        return "(no prior context)"
    # Optionally truncate or summarize context for the LLM
    return context[-500:]

async def infer_posture_from_message_llm(message: str, context: Optional[str] = None) -> Optional[str]:
    """
    Use the LLM to infer the user's posture from their message and recent context.
    Returns a single word or short phrase, or None if not detected.
    """
    context_snippet = _get_recent_context_snippet(context)
    prompt = POSTURE_PROMPT.format(context=context_snippet, message=message)
    try:
        llm = get_llm_provider()
        response = await llm.generate(prompt=prompt, max_tokens=settings.VLLM_MAX_TOKENS, temperature=settings.VLLM_TEMPERATURE)
        content = response.strip().lower()
        for posture in POSTURE_TYPES:
            if posture in content:
                return posture
        return content if content else None
    except Exception as e:
        logger.error(f"Posture LLM inference failed: {e}", exc_info=True)
        return None 