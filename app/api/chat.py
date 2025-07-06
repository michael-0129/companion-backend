"""
This module defines the main agent interaction endpoint for the Companion API.
It orchestrates the process of receiving a user query, classifying its intent,
dispatching it to the appropriate services via the TaskOrchestrator,
and then persisting the interaction to the chat history.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
import json
from uuid import UUID
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
import re
from fastapi.responses import JSONResponse

from app import schemas
from app.db.session import get_db
from app.core.config import settings, TIMEZONE
from app.core.logging_config import get_logger
from app.core.exceptions import CoreApplicationException, InputTooLongError
from typing import Dict, Any
from app.services import chat
from app.services.protocol import (
    get_active_protocol_event,
    deactivate_protocol_event
)
from app.services.chat import create_chat_history
from app.services.chat import get_chat_history as get_chat_history_service, generate_and_save_chat_summary
from app.services.task_orchestrator import TaskOrchestrator
from app.utils.llm_context_v2 import LlmContextManager
from app.utils.llm_provider import get_llm_provider
from app.prompts.system_prompts import SYSTEM_PROMPT_CLASSIFY
from app.utils.token_counter import TokenCounter

router = APIRouter()
logger = get_logger(__name__)

cet_tz = ZoneInfo(TIMEZONE)

def extract_json_from_code_block(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

async def _handle_silence_protocol(db: Session) -> bool:
    """
    Checks if the silence protocol is currently active. If an expired protocol
    is found, it is deactivated.
    """
    active_silence_protocol = get_active_protocol_event(db, event_type="silence_mode")
    if not active_silence_protocol:
        return False

    active_until_str = active_silence_protocol.details.get("active_until")
    if not active_until_str:
        logger.info(f"Silence mode is active indefinitely (Protocol ID: {active_silence_protocol.id}).")
        return True

    try:
        active_until_dt = datetime.fromisoformat(active_until_str)
        if datetime.now(cet_tz) < active_until_dt:
            logger.info(f"Silence mode is active until {active_until_dt.isoformat()} (Protocol ID: {active_silence_protocol.id}).")
            return True
        else:
            logger.info(f"Silence mode protocol {active_silence_protocol.id} has expired. Deactivating.")
            deactivate_protocol_event(db, active_silence_protocol.id)
            return False
    except (ValueError, TypeError):
        logger.error(f"Invalid 'active_until' format ('{active_until_str}') in protocol {active_silence_protocol.id}. Assuming not silent.")
        return False

@router.post("/agent", response_model=schemas.ChatHistoryOut)
async def agent_interaction(
    request: schemas.AgentInteractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Main endpoint for all user interactions with the agent.

    This endpoint performs the following steps:
    1. Checks for an active "silence mode" protocol.
    2. Uses an LLM to classify the user's query into a specific intent.
    3. Initializes and runs the TaskOrchestrator to execute the intent.
    4. Catches any application-level errors and converts them to HTTPExceptions.
    5. Persists the full interaction (query, response, context) to the database.
    """
    user_query = request.user_query
    # Fetch the current active directive (if any)
    active_directive_event = get_active_protocol_event(db, event_type="directive")
    directive = None
    if active_directive_event and active_directive_event.details:
        directive = active_directive_event.details.get("directive_content")
    # Build protocol block if directive exists
    protocol_block = ""
    if directive:
        protocol_block = directive
    # Early tokenization check: include system prompt + protocol block + user query
    system_prompt = SYSTEM_PROMPT_CLASSIFY.format(
        user_query=user_query,
        current_date=datetime.now(cet_tz).strftime("%Y-%m-%d"),
        protocol_block=protocol_block
    )
    token_counter = TokenCounter(settings.VLLM_MODEL)
    total_tokens = token_counter.count(system_prompt) + token_counter.count(user_query)
    if total_tokens > settings.VLLM_MAX_INPUT_TOKENS:
        logger.warning(f"Classification prompt + user query exceed max input tokens ({total_tokens} > {settings.VLLM_MAX_INPUT_TOKENS})")
        return JSONResponse(
            status_code=422,
            content={"error": f"Your query and prompt exceed the maximum allowed input size ({settings.VLLM_MAX_INPUT_TOKENS} tokens). Please shorten your request."}
        )
    # Central context object for logging and data passing throughout the request.
    context_snapshot: Dict[str, Any] = {}
    
    try:
        # Step 1: Handle Protocols (e.g., Silence Mode)
        silence_effectively_active = await _handle_silence_protocol(db)
        context_snapshot["initial_silence_state"] = silence_effectively_active

        # Step 2: Intent Classification
        classification_manager = LlmContextManager(
            model_name=settings.VLLM_MODEL, 
            max_input_tokens=settings.VLLM_MAX_INPUT_TOKENS,
            max_output_tokens=settings.VLLM_MAX_OUTPUT_TOKENS
        )
        classification_manager.set_system_prompt(system_prompt)
        classification_manager.add_query(user_query)
        messages = classification_manager.get_classify_messages()
        logger.info(f"Classification prompt sent to LLM: {system_prompt}")
        llm = get_llm_provider()
        classify_response = await llm.generate(
            messages=messages,
            max_tokens=settings.VLLM_MAX_OUTPUT_TOKENS,
            temperature=0.1
        )
        logger.info(f"Raw classify_response from LLM: '{classify_response}'")
        if not classify_response or not classify_response.strip():
            logger.error("LLM classification returned empty response. Returning protocol-aligned error message.")
            return schemas.ChatHistoryOut(
                id=None,
                user_query=user_query,
                companion_response="I encountered an internal error during classification. Please try again later.",
                created_at=datetime.now(cet_tz),
                protocol_blocked=True
            )
        cleaned_response = extract_json_from_code_block(classify_response)
        classification_data = json.loads(cleaned_response)
        logger.info(f"LLM classification output: {classify_response}")
        # --- Robustly handle both dict and list outputs from LLM ---
        if isinstance(classification_data, list):
            tasks = classification_data
        elif isinstance(classification_data, dict):
            if "tasks" in classification_data and isinstance(classification_data["tasks"], list):
                tasks = classification_data["tasks"]
            else:
                tasks = [classification_data]
        else:
            logger.error(f"Invalid classification_data format: {type(classification_data)}")
            raise ValueError("Invalid classification_data format from LLM")
        context_snapshot["classification_data"] = classification_data
        
        # Step 3: Task Orchestration
        orchestrator = TaskOrchestrator(
            user_query=user_query,
            initial_classification_data=classification_data,
            db=db,
            context_snapshot=context_snapshot,
            tasks=tasks  # Pass the robustly parsed tasks list
        )
        
        # The orchestrator is now the single point of entry for all handler logic.
        # --- Handle InputTooLongError from RAG/QUERY context assembly ---
        try:
            companion_response_content, linked_codex_id, orchestration_errors = await orchestrator.execute_plan()
        except InputTooLongError as e:
            logger.warning(f"Context for answer exceeds max input tokens: {e}")
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
        if orchestration_errors:
            context_snapshot['llm_call_error'] = orchestration_errors
        
        # If silence mode is active, override any generated response,
        # UNLESS this is a COMMAND (so user gets confirmation)
        is_command = False
        if isinstance(classification_data, dict):
            tasks = classification_data.get("tasks")
            if tasks and isinstance(tasks, list):
                for task in tasks:
                    if task.get("intent") == "COMMAND":
                        is_command = True
                        break
            elif classification_data.get("intent") == "COMMAND":
                is_command = True

        if silence_effectively_active and not is_command:
            companion_response_content = settings.SILENCE_MODE_RESPONSE

    except InputTooLongError as e:
        logger.warning(f"Input too long: {e}")
        return JSONResponse(
            status_code=422,
            content={"error": str(e)}
        )
    except CoreApplicationException as e:
        # This is the catch-all for our custom, service-level exceptions.
        # It ensures that a well-defined error from a service is translated
        # into a user-facing HTTP 500 error without crashing the application.
        logger.error(f"A core application error occurred: {e.message}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A service error occurred: {e.message}")
    except (json.JSONDecodeError, ValueError) as e:
        # Handles errors if the classification LLM returns malformed JSON.
        logger.error(f"Failed to decode or parse classification JSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse LLM response for classification.")
    except Exception as e:
        # This is a final safeguard for any other unexpected errors.
        logger.critical(f"An unhandled critical error occurred in agent interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected critical error occurred.")

    # Step 4: Persist Interaction to Chat History
    # This runs regardless of success or failure (unless an HTTPException was raised).
    # --- Protocol Alignment: Do not persist blocked memory attempts to chat history ---
    if "protocol_blocked_memory" in context_snapshot:
        logger.info("Blocked memory attempt detected; saving protocol block to chat history.")
        chat_history_data = schemas.ChatHistoryCreate(
            user_query=user_query,
            companion_response=companion_response_content,  # The protocol block message
            context_snapshot=context_snapshot,
            linked_codex_entry=None
        )
        chat_history_entry = create_chat_history(db=db, entry=chat_history_data)
        return chat_history_entry
    # Otherwise, persist as normal
    chat_history_data = schemas.ChatHistoryCreate(
        user_query=user_query,
        companion_response=companion_response_content,
        context_snapshot=context_snapshot,
        linked_codex_entry=linked_codex_id
    )
    chat_history_entry = create_chat_history(db=db, entry=chat_history_data)
    # Trigger summary generation in the background
    background_tasks.add_task(generate_and_save_chat_summary, db, chat_history_entry)
    logger.info(f"Background summary generation task added for ChatHistory {chat_history_entry.id}.")
    # Return the persisted chat history entry, confirming the interaction is complete.
    return chat_history_entry

@router.get("/history", response_model=List[schemas.ChatHistoryOut])
async def get_chat_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_query: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get chat history with optional filtering by date range and search query.
    Results are paginated.
    """
    try:
        return await get_chat_history_service(
            db=db,
            skip=skip,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            search_query=search_query
        )
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{chat_id}", response_model=schemas.ChatHistoryOut)
async def get_chat_entry(
    chat_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific chat history entry by ID.
    """
    chat_entry = await chat.get_chat_entry(db=db, chat_id=chat_id)
    if not chat_entry:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    return chat_entry

@router.delete("/history/{chat_id}")
async def delete_chat_entry(
    chat_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a specific chat history entry.
    """
    success = await chat.delete_chat_entry(db=db, chat_id=chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    return {"status": "success", "message": "Chat entry deleted successfully"}