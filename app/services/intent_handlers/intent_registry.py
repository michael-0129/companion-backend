"""Intent handler registry and dispatcher.

This module defines the mapping from intent strings (e.g., "MEMORY", "COMMAND")
to their corresponding handler functions using a decorator-based registration system.
It provides an `IntentRegistry` class instance that handlers can use to register themselves.
"""
from typing import Callable, Dict, Any, Coroutine, Optional
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Define a type alias for the handler functions for clarity.
# Handler functions are asynchronous and take a variable set of arguments
# (db session, user query, classification data, context snapshot, etc.)
# and are expected to return a tuple, though the exact structure of the tuple
# can vary. The main `agent_interaction` loop in `chat.py` currently handles
# unpacking these different return signatures based on which handler was called.
# A more unified return object (e.g., a Pydantic model or dataclass) might be considered
# in the future if handler signatures diverge significantly or become too complex to manage.
IntentHandler = Callable[..., Coroutine[Any, Any, Any]] # Represents a generic asynchronous callable (the intent handler function).

async def handle_unknown_intent(
    db: Session,
    user_query: str,
    classification_data: Dict[str, Any],
    context_snapshot: Dict[str, Any],
    silence_effectively_active: bool,
    current_llm_call_error: Optional[str]
) -> tuple[str, Optional[UUID], Optional[str]]:
    """
    Fallback handler for intents not explicitly found in the registry.

    This handler is invoked when `get_handler` cannot find a specific handler.
    """
    intent = classification_data.get("intent", "UNCLASSIFIED_FALLBACK")
    logger.error(f"Fallback handler 'handle_unknown_intent' invoked for intent: '{intent}'. Query: '{user_query[:100]}...'.")
    
    response_content = "I'm not sure how to handle that type of request. This has been logged for review."
    if silence_effectively_active:
        logger.info(f"Silence mode active. Suppressing response for unhandled intent '{intent}'.")
        response_content = "" # Standard silence mode response might be better, but for unknown, empty is safer.
        
    error_message = current_llm_call_error or f"Unhandled intent: '{intent}' was processed by the fallback handler."
    
    # Update context snapshot with details about this unhandled intent.
    context_snapshot["unhandled_intent_details"] = {
        "original_intent": intent,
        "user_query_snippet": user_query[:200],
        "message": "Processed by fallback 'handle_unknown_intent'.",
        "classification_data_snippet": {k: str(v)[:100] + '...' if isinstance(v, (str, list, dict)) and len(str(v)) > 100 else v for k, v in classification_data.items()} # Snippet of classification data
    }
    return response_content, None, error_message

class IntentRegistry:
    """A registry for intent handler functions."""
    def __init__(self):
        self._registry: Dict[str, IntentHandler] = {}
        # Set the fallback handler explicitly. It's defined before this class.
        self._registry["UNKNOWN"] = handle_unknown_intent

    def register(self, intent_name: str) -> Callable[[IntentHandler], IntentHandler]:
        """Returns a decorator that registers an intent handler function."""
        def decorator(handler_func: IntentHandler) -> IntentHandler:
            logger.debug(f"Registering handler for intent: '{intent_name}'")
            if intent_name in self._registry:
                logger.warning(f"Intent '{intent_name}' is already registered. Overwriting with new handler: {handler_func.__name__}")
            self._registry[intent_name] = handler_func
            return handler_func
        return decorator

    def get_handler(self, intent_name: str) -> IntentHandler:
        """Retrieves the handler for a given intent, or the fallback."""
        return self._registry.get(intent_name, self._registry["UNKNOWN"])

    def get_fallback_handler(self) -> IntentHandler:
        """Explicitly gets the fallback handler."""
        return self._registry["UNKNOWN"]

intent_handler_registry = IntentRegistry()

def get_intent_handler(intent: str) -> IntentHandler:
    """
    Retrieves the appropriate handler function for a given intent string.

    If the intent string is found in the `intent_handler_registry`, the corresponding
    handler function is returned. If the intent is not found, this function returns
    the `handle_unknown_intent` function as a fallback mechanism.

    Args:
        intent: The intent string (e.g., "MEMORY", "QUERY") classified by the LLM.

    Returns:
        The callable handler function corresponding to the intent, or the
        `handle_unknown_intent` if the intent is not registered.
    """
    handler = intent_handler_registry.get_handler(intent)
    return handler