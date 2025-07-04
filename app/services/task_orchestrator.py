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
from zoneinfo import ZoneInfo
from app.core.config import TIMEZONE

from app.core.logging_config import get_logger
from app.core.exceptions import TaskOrchestrationError
from app.services.intent_handlers.intent_registry import intent_handler_registry, IntentHandler
logger = get_logger(__name__)

CET_TZ = ZoneInfo(TIMEZONE)  # Use your canonical CET timezone

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
        try:
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
                # Only add non-empty, non-duplicate 'Received, Michael.' responses
                if response_content:
                    user_responses.append(response_content)
                if linked_codex_id:
                    linked_codex_ids.append(linked_codex_id)
                if handler_errors:
                    self.llm_call_errors.append(str(handler_errors))
            final_errors = "; ".join(self.llm_call_errors) if self.llm_call_errors else None
            # --- Minimal confirmation logic ---
            # If any response is a protocol block or error, return as is
            protocol_block = any(
                "protocol_blocked_memory" in self.context_snapshot or
                (resp and ("sealed" in resp or "No new memory may be recorded" in resp))
                for resp in user_responses
            )
            if protocol_block:
                return user_responses[0], (linked_codex_ids[0] if linked_codex_ids else None), final_errors
            # If there are any user_responses, deduplicate and only return a single minimal confirmation
            if user_responses:
                # If any response is not exactly 'Received, Michael.' or is a protocol block, return it
                for resp in user_responses:
                    if resp.strip() != "Received, Michael.":
                        return resp, (linked_codex_ids[0] if linked_codex_ids else None), final_errors
                # Otherwise, return a single minimal confirmation
                return "Received, Michael.", (linked_codex_ids[0] if linked_codex_ids else None), final_errors
            # If there are no QUERY/COMMAND responses, but a handler created an object (e.g., memory), return a protocol-aligned confirmation
            if not user_responses and linked_codex_ids:
                return "Received, Michael.", linked_codex_ids[0], final_errors
            return "", (linked_codex_ids[0] if linked_codex_ids else None), final_errors
        except TaskOrchestrationError as e:
            logger.error(f"A recoverable error occurred during task orchestration: {e.message}", exc_info=True)
            self.llm_call_errors.append(e.message)
            return "I encountered an issue while processing your request.", None, "; ".join(self.llm_call_errors)
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred in the orchestrator: {str(e)}", exc_info=True)
            error_msg = f"An unexpected critical error occurred in the task orchestrator: {str(e)}"
            self.llm_call_errors.append(error_msg)
            return "A critical and unexpected error occurred.", None, "; ".join(self.llm_call_errors)