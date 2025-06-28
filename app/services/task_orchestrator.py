"""
The Task Orchestrator is responsible for executing a plan to respond to a user's query.
It takes the initial classification from the `chat.py` endpoint and uses the 
intent handler registry to call the appropriate service function.

For now, this orchestrator is simple: it directly maps one intent to one handler call.
In the future, it could be extended to handle multi-step tasks defined by an LLM plan.
"""
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from uuid import UUID
import json
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.logging_config import get_logger
from app.core.exceptions import TaskOrchestrationError
# Correctly import the registry instance and the handler type definition
from app.services.intent_handlers.intent_registry import intent_handler_registry, IntentHandler
from app.utils.posture import infer_posture_from_message_llm
from app.services.codex_service import create_codex_entry
from app import schemas
from app.models import CodexEntry
from app.utils.intent import classify_intent_from_message_llm

logger = get_logger(__name__)

CET_TZ = ZoneInfo("Europe/Paris")  # Use your canonical CET timezone

class TaskOrchestrator:
    """
    Orchestrates the execution of tasks based on an initial intent classification.
    """
    def __init__(self, 
                 user_query: str, 
                 initial_classification_data: Dict[str, Any],
                 db: Session,
                 context_snapshot: Dict[str, Any],
                 tasks: list = None):
        """
        Initializes the TaskOrchestrator.

        Args:
            user_query: The original query from the user.
            initial_classification_data: The JSON object from the first classification LLM call.
            db: The SQLAlchemy database session.
            context_snapshot: A dictionary to log metadata about the execution.
            tasks: (Optional) List of tasks to execute. If not provided, will be parsed from classification_data for legacy support.
        """
        self.user_query = user_query
        self.classification_data = initial_classification_data
        self.db = db
        self.context_snapshot = context_snapshot
        self.llm_call_errors: list[str] = []
        self.registry = intent_handler_registry
        # Use provided tasks if available, else parse from classification_data (legacy support)
        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = self.classification_data.get("tasks")
            if not self.tasks:
                # Fallback: treat as single task for backward compatibility
                self.tasks = [{
                    "intent": self.classification_data.get("intent"),
                    "parameters": self.classification_data
                }]
        logger.info(f"TaskOrchestrator initialized for {len(self.tasks)} task(s).")

    async def execute_plan(self) -> Tuple[str, Optional[UUID], Optional[str]]:
        """
        Executes all tasks in the plan. Aggregates user-facing responses.
        Returns:
            A tuple containing the aggregated string response, an optional linked Codex ID (first one), and any accumulated error messages.
        """
        user_responses = []
        linked_codex_ids = []
        posture_mirroring_phrase = None
        try:
            # --- Intent Classification: Decide if posture tracking is needed ---
            intent_class = await classify_intent_from_message_llm(self.user_query)
            logger.info(f"[ORCHESTRATOR] Intent classification for user_query: '{self.user_query}' => '{intent_class}'")
            current_posture = None
            previous_posture = None
            posture_codex_id = None
            # Only infer posture if intent is inner_state or multi_intent (never for greeting)
            if intent_class in ("inner_state", "multi_intent"):
                current_posture = await infer_posture_from_message_llm(self.user_query)
                # Query the most recent posture_state CodexEntry
                recent_posture_entry = (
                    self.db.query(CodexEntry)
                    .filter(CodexEntry.type == 'posture_state')
                    .order_by(CodexEntry.created_at.desc())
                    .first()
                )
                if recent_posture_entry and hasattr(recent_posture_entry, 'meta'):
                    meta = recent_posture_entry.meta or {}
                    previous_posture = meta.get('current_posture')
                # If a posture shift is detected, create a posture memory and prepare mirroring phrase
                if current_posture and previous_posture and current_posture != previous_posture:
                    posture_content = f"Posture shift detected: {previous_posture} → {current_posture}"
                    posture_tags = ["posture", previous_posture, current_posture]
                    posture_meta = {
                        "user_query": self.user_query,
                        "previous_posture": previous_posture,
                        "current_posture": current_posture,
                        "detected_at": datetime.now(CET_TZ).isoformat(),
                    }
                    posture_entry = schemas.CodexEntryCreate(
                        content=posture_content,
                        tags=posture_tags,
                        entities=[],
                        meta=posture_meta,
                        archived=False,
                        type="posture_state",
                        protocol_flags=[],
                        event_date=datetime.now(CET_TZ).date(),
                    )
                    codex_obj = await create_codex_entry(self.db, posture_entry, embedding=None)
                    posture_codex_id = codex_obj.id
                    posture_mirroring_phrase = (
                        f"I sense a gentle shift in your focus, Michael—from {previous_posture} to {current_posture}. "
                    )
                elif current_posture and not previous_posture:
                    posture_content = f"Initial posture detected: {current_posture}"
                    posture_tags = ["posture", current_posture]
                    posture_meta = {
                        "user_query": self.user_query,
                        "previous_posture": None,
                        "current_posture": current_posture,
                        "detected_at": datetime.now(CET_TZ).isoformat(),
                    }
                    posture_entry = schemas.CodexEntryCreate(
                        content=posture_content,
                        tags=posture_tags,
                        entities=[],
                        meta=posture_meta,
                        archived=False,
                        type="posture_state",
                        protocol_flags=[],
                        event_date=datetime.now(CET_TZ).date(),
                    )
                    codex_obj = await create_codex_entry(self.db, posture_entry, embedding=None)
                    posture_codex_id = codex_obj.id
                    posture_mirroring_phrase = None  # No mirroring for first detection
            elif intent_class == "greeting":
                logger.info(f"[ORCHESTRATOR] Greeting detected, skipping posture inference and mirroring.")
            else:
                logger.info(f"[ORCHESTRATOR] Skipping posture inference for intent_class '{intent_class}'")
            for idx, task in enumerate(self.tasks):
                intent = task.get("intent")
                parameters = task  # Pass the full task dict
                if not intent:
                    raise TaskOrchestrationError(f"Intent not found in task #{idx+1}.")
                # --- Backend validation/interceptor for archive vs silence mode ---
                if intent == "COMMAND":
                    cmd_name = parameters.get("command_name")
                    user_query_lower = self.user_query.lower()
                    if (
                        cmd_name == "SET_SILENCE_MODE"
                        and "archive" in user_query_lower
                    ):
                        logger.warning(f"Intercepted likely LLM misclassification: user_query contains 'archive' but command_name is 'SET_SILENCE_MODE'. Overriding to 'SET_ARCHIVE_MODE'.")
                        self.tasks[idx]["command_name"] = "SET_ARCHIVE_MODE"
                        if "parameters" in self.tasks[idx] and isinstance(self.tasks[idx]["parameters"], dict):
                            self.tasks[idx]["parameters"]["command_name"] = "SET_ARCHIVE_MODE"
                        logger.info(f"Patched self.tasks[{idx}] for command_name: {self.tasks[idx]}")
                # --- End backend validation/interceptor ---
                # Log the actual object passed to the handler
                logger.info(f"[ORCHESTRATOR] Passing to handler: {self.tasks[idx]}")
                handler: IntentHandler = self.registry.get_handler(intent)
                logger.info(f"Executing handler '{handler.__name__}' for intent '{intent}' (task #{idx+1}).")
                response_content, linked_codex_id, handler_errors = await handler(
                    db=self.db,
                    user_query=self.user_query,
                    classification_data=self.tasks[idx],
                    context_snapshot=self.context_snapshot,
                    silence_effectively_active=self.context_snapshot.get("initial_silence_state", False),
                    current_llm_call_error=None
                )
                logger.info(f"Handler result: response_content='{response_content}', linked_codex_id={linked_codex_id}, handler_errors={handler_errors}")
                # --- Protocol block detection ---
                if (
                    response_content and
                    linked_codex_id is None and
                    "protocol_blocked_memory" in self.context_snapshot
                ):
                    logger.info("Protocol block detected in handler result. Returning protocol message to user and aborting further task execution.")
                    return response_content, None, handler_errors
                # Include any non-empty response_content in the user-facing output
                if response_content:
                    user_responses.append(response_content)
                if linked_codex_id:
                    linked_codex_ids.append(linked_codex_id)
                if handler_errors:
                    self.llm_call_errors.append(str(handler_errors))
            final_errors = "; ".join(self.llm_call_errors) if self.llm_call_errors else None
            # If there are no QUERY/COMMAND responses, but a handler created an object (e.g., memory), return a protocol-aligned confirmation
            if not user_responses and linked_codex_ids:
                # Try to extract more context for a richer message
                item_type = None
                if self.tasks and isinstance(self.tasks[0], dict):
                    intent = self.tasks[0].get('intent')
                    params = self.tasks[0].get('parameters', self.tasks[0])
                    if intent == 'MEMORY':
                        item_type = params.get('memory_type', 'memory')
                    elif intent == 'FORCED_ARCHIVE_MEMORY':
                        item_type = 'archived memory'
                    elif intent == 'COMMAND':
                        item_type = 'command result'
                    elif intent == 'DOCUMENT':
                        item_type = 'document'
                    if 'filename' in params:
                        user_responses.append(f"Received, Michael. The document '{params['filename']}' has been integrated.")
                    else:
                        user_responses.append(f"Received, Michael. The requested {item_type or 'item'} has been integrated.")
                else:
                    user_responses.append("Received, Michael. The requested item has been integrated.");
            # --- Posture Mirroring: Prepend phrase if posture shift detected ---
            aggregated_response = "\n".join([resp for resp in user_responses if resp])
            if posture_mirroring_phrase and intent_class in ("inner_state", "multi_intent"):
                aggregated_response = posture_mirroring_phrase + aggregated_response

            # --- Mythic Posture Affirmation: Replace generic memory confirmation if posture-only ---
            # If the only response is a generic memory confirmation for a posture/realization event, replace it
            generic_memory_phrases = [
                "The memory has been integrated and its essence is preserved.",
                "Memory recorded. Its essence is preserved.",
                "Received, Michael. The memory has been integrated and its essence is preserved. (Type: 'realization')"
            ]
            # Simple mapping for posture types to mythic affirmations
            posture_affirmations = {
                "clarity": "Clarity is a powerful threshold, Michael. Trust what has emerged, and let it guide your next step.",
                "confusion": "Confusion is an invitation, not an obstacle. Remain present, and let the next step emerge.",
                "curiosity": "Curiosity is the compass of discovery. Let it lead you into new understanding.",
                "hesitation": "Hesitation is a pause before transformation. Honor it, and move when ready.",
                "confidence": "Confidence is the mark of readiness. Step forward with assurance, Michael.",
                "reflection": "Reflection deepens wisdom. Let your insights shape the path ahead.",
                "uncertainty": "Uncertainty is the edge of growth. Trust the process, and clarity will follow."
            }
            # If only one response and it's a generic memory confirmation, replace it
            if len(user_responses) == 1:
                resp = user_responses[0]
                for phrase in generic_memory_phrases:
                    if phrase in resp or resp.strip().startswith("Received, Michael. The memory has been integrated"):
                        # Try to extract posture from current_posture
                        mythic_phrase = posture_affirmations.get(current_posture, "A new state has emerged, Michael. Let it guide you.")
                        aggregated_response = (posture_mirroring_phrase or "") + mythic_phrase
                        break
            first_codex_id = linked_codex_ids[0] if linked_codex_ids else None
            logger.info(f"All tasks executed successfully.")
            return aggregated_response, first_codex_id, final_errors
        except TaskOrchestrationError as e:
            logger.error(f"A recoverable error occurred during task orchestration: {e.message}", exc_info=True)
            self.llm_call_errors.append(e.message)
            return "I encountered an issue while processing your request.", None, "; ".join(self.llm_call_errors)
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred in the orchestrator: {str(e)}", exc_info=True)
            error_msg = f"An unexpected critical error occurred in the task orchestrator: {str(e)}"
            self.llm_call_errors.append(error_msg)
            return "A critical and unexpected error occurred.", None, "; ".join(self.llm_call_errors)