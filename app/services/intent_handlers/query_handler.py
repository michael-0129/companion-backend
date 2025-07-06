"""
Handler for processing query-like intents using a Retrieval Augmented Generation (RAG) pipeline.

This module is responsible for intents classified as "QUERY", "UNKNOWN", or "COMPLEX_TASK".
It fetches relevant information from memory (Codex entries) and recent chat history,
constructs a comprehensive prompt, and then calls an LLM to generate a response to the user's query.
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from zoneinfo import ZoneInfo

# schemas might be used if we were creating/returning structured data, not typical for query response.
# from app import schemas, models 
from app.core.config import settings, TIMEZONE
from app.core.logging_config import get_logger
from app.services.memory import (
    semantic_search_codex,
    DatabaseOperationError,
    DocumentProcessingError        
)
from app.services.chat import get_chat_history as get_chat_history_service
from app.services.intent_handlers.intent_registry import intent_handler_registry
from app.prompts.system_prompts import SYSTEM_PROMPT_ANSWER
from app.utils.security import decrypt_content
from app.services import relational_state  # Import relational state for closure event lookup
from app.utils.llm_provider import get_llm_provider
from app.services.chat import generate_context_block_summary
from app.utils.token_counter import TokenCounter
from app.core.exceptions import InputTooLongError
from app.services.protocol import list_protocol_events

logger = get_logger(__name__)

cet_tz = ZoneInfo(TIMEZONE)

@intent_handler_registry.register("QUERY")
@intent_handler_registry.register("COMPLEX_TASK")
async def handle_query_intent(
    db: Session,
    user_query: str,
    classification_data: Dict[str, Any],
    context_snapshot: Dict[str, Any], 
    silence_effectively_active: bool,
    current_llm_call_error: Optional[str],
    start_date_param: Optional[str] = None,
    end_date_param: Optional[str] = None
) -> Tuple[str, Optional[UUID], Optional[str]]:
    """
    Handles "QUERY", "UNKNOWN", and "COMPLEX_TASK" intents using a RAG pipeline.
    Now supports both legacy and new multi-task formats (parameters dict).

    The process involves:
    1. Performing a semantic search for relevant codex entries (memories).
    2. Retrieving recent chat history.
    3. Using `LlmContextManager` to assemble a prompt with the user query, retrieved memories, and chat history.
    4. Calling an OpenAI LLM (via `get_openai_completion`) to generate the final answer.
    5. If silence mode is active, the RAG process is skipped, and an empty response is returned.

    Args:
        db: The SQLAlchemy database session.
        user_query: The original user query.
        classification_data: Dictionary from intent classification. It may contain
                             `summary_for_query` to be used for semantic search.
        context_snapshot: Dictionary for logging metadata about the RAG process (e.g., number of retrieved items).
        silence_effectively_active: Boolean indicating if silence mode is active.
        current_llm_call_error: Optional string of accumulated error messages from prior steps.
        start_date_param: Optional start date string (YYYY-MM-DD) for filtering semantic search.
        end_date_param: Optional end date string (YYYY-MM-DD) for filtering semantic search.

    Returns:
        A tuple containing:
        - companion_response_content (str): The LLM-generated answer to the user's query.
        - linked_codex_id (Optional[UUID]): Always None for query intents, as they don't create a new primary codex entry.
        - llm_call_error_updated (Optional[str]): Updated string of accumulated error messages.

    Raises:
        This function catches specific custom exceptions from called services (`semantic_search_codex`, `get_chat_history_service`)
        and `HTTPException` from `get_openai_completion`. It aims to handle them by logging and returning an error message to the user.
        Expected exceptions from services include:
        - `OpenAIClientNotInitializedError`, `OpenAIAPICallError`, `DocumentProcessingError` (from `semantic_search_codex` if embedding fails).
        - `DatabaseOperationError` (from `semantic_search_codex` or `get_chat_history_service`).
    """
    # Accept parameters dict if present (from orchestrator multi-task), else fallback to classification_data
    parameters = classification_data.get("parameters") if "parameters" in classification_data else classification_data
    intent = parameters.get("intent", classification_data.get("intent", "QUERY"))
    logger.info(f"Handling intent: '{intent}' in query_handler for user query: '{user_query[:100]}...'. Preparing for RAG.")

    # --- Relational State Enforcement for Query ---
    # If the query is about a closed field, only use memories up to the closure event
    referenced_fields = []
    if "extracted_entities" in parameters:
        referenced_fields = [e["text"] for e in parameters["extracted_entities"] if e.get("type") == "RELATIONAL_FIELD"]
    closed_fields = relational_state.get_closed_fields(db)
    closure_dates = {}
    for field in referenced_fields:
        if field in closed_fields:
            # Find the closure event date for this field
            entries = db.query(db.registry.mapped['CodexEntry']).filter(
                db.registry.mapped['CodexEntry'].type == 'relational_state',
                db.registry.mapped['CodexEntry'].tags.any('closed_fields'),
                db.registry.mapped['CodexEntry'].entities.any({"text": field, "type": "RELATIONAL_FIELD"}),
                db.registry.mapped['CodexEntry'].meta["closed"].astext == 'true'
            ).order_by(db.registry.mapped['CodexEntry'].created_at.desc()).all()
            if entries:
                # Ensure closure_date is timezone-aware and in cet_tz
                closure_date = entries[0].event_date
                if closure_date and closure_date.tzinfo is None:
                    closure_date = datetime.combine(closure_date, datetime.min.time(), cet_tz)
                closure_dates[field] = closure_date
    # ... existing code ...
    companion_response_content = ""
    llm_call_error_updated = current_llm_call_error

    if silence_effectively_active:
        logger.info(f"Silence mode is active. Skipping RAG pipeline and response generation for intent: '{intent}'.")
        return "", None, llm_call_error_updated

    # Use summary from parameters for search if available, otherwise use the raw user query.
    query_summary_for_search = parameters.get("summary_for_query", user_query)
    if not query_summary_for_search or not query_summary_for_search.strip(): 
        logger.warning(f"summary_for_query from parameters was empty for intent '{intent}', using original user_query for semantic search.")
        query_summary_for_search = user_query

    # Use event_date or date filters if present in parameters
    start_date_param = parameters.get("start_date") or parameters.get("event_date") or start_date_param
    end_date_param = parameters.get("end_date") or end_date_param

    # --- Directive Enforcement ---
    # Fetch the latest active directive (if any)
    latest_directive = None
    directives = list_protocol_events(db, event_type="directive", active=True)
    if directives:
        # Assume the most recent is first (ordered by timestamp desc)
        latest_directive = directives[0].details.get("directive_content")

    # Prepare SYSTEM_PROMPT_ANSWER with directive injected at the top
    base_system_prompt = SYSTEM_PROMPT_ANSWER
    if latest_directive:
        # Truncate directive if too long (e.g., >512 chars)
        max_directive_chars = 512
        directive_for_prompt = latest_directive[:max_directive_chars]
        system_prompt_for_answer = f"{directive_for_prompt}\n\n{base_system_prompt}"
    else:
        system_prompt_for_answer = base_system_prompt

    try:
        # Step 1: Semantic Search for relevant memories (Codex Entries)
        logger.debug(f"Performing semantic search for RAG query: '{query_summary_for_search[:100]}...', start_date: {start_date_param}, end_date: {end_date_param}")
        retrieved_memories = await semantic_search_codex(
            db, 
            query_text=query_summary_for_search, 
            top_k=settings.MAX_SEMANTIC_SEARCH_RESULTS,
            start_date_str=start_date_param, 
            end_date_str=end_date_param     
        )
        # --- Filter memories for closed fields ---
        if closure_dates:
            filtered_memories = []
            for m in retrieved_memories:
                for field, closure_date in closure_dates.items():
                    # Ensure memory event_date is timezone-aware and in cet_tz
                    mem_date = m.event_date
                    if mem_date and mem_date.tzinfo is None:
                        mem_date = datetime.combine(mem_date, datetime.min.time(), cet_tz)
                    if field in m.content and mem_date and mem_date > closure_date:
                        logger.info(f"Excluding memory for closed field '{field}' after closure date {closure_date}.")
                        break
                else:
                    filtered_memories.append(m)
            retrieved_memories = filtered_memories
        # --- End filter ---
        context_snapshot["retrieved_memories_count"] = len(retrieved_memories)
        context_snapshot["semantic_search_query_used"] = query_summary_for_search
        logger.info(f"Retrieved {len(retrieved_memories)} memories from semantic search for RAG pipeline.")

        # Step 2: Retrieve recent chat history
        recent_chat_history = await get_chat_history_service(db, limit=settings.MAX_RECENT_CHAT_HISTORY)
        context_snapshot["recent_chat_history_turns_retrieved"] = len(recent_chat_history)
        logger.info(f"Retrieved {len(recent_chat_history)} recent chat turns for RAG context.")

        # Prepare memory strings for context summary
        memory_summaries = [
            f"Memory from {m.event_date.strftime('%Y-%m-%d') if m.event_date else 'Unknown date'}: {decrypt_content(m.encrypted_content)}"
            for m in retrieved_memories
        ]

        # Use summaries if available, else fallback to full turn
        chat_summaries = [
            c.summary if getattr(c, 'summary', None) else f"User: {c.user_query}\nAI: {c.companion_response}"
            for c in recent_chat_history
        ]

        # --- Context condensation for token safety ---
        max_context_tokens = settings.VLLM_MAX_INPUT_TOKENS // 2  # or another safe budget
        context_block_summary = await generate_context_block_summary(
            user_query=user_query,
            chat_summaries=chat_summaries,
            memories=memory_summaries,
            max_tokens=max_context_tokens
        )

        # Use the condensed context block in the prompt
        messages = [
            {"role": "system", "content": system_prompt_for_answer},
            {"role": "user", "content": f"{context_block_summary}\n\n{user_query}"}
        ]
        token_counter = TokenCounter(settings.VLLM_MODEL)
        total_tokens = token_counter.count_messages(messages)
        if total_tokens > settings.VLLM_MAX_INPUT_TOKENS:
            logger.warning(f"Final assembled prompt exceeds input token limit ({total_tokens} > {settings.VLLM_MAX_INPUT_TOKENS}) even after summarization.")
            raise InputTooLongError(f"Context cannot be reduced below input token limit ({settings.VLLM_MAX_INPUT_TOKENS})!")
        logger.debug(f"Assembled prompt for {settings.VLLM_MODEL} (dynamic context). Messages: {messages}")
        llm = get_llm_provider()
        answer_response = await llm.generate(
            messages=messages,
            max_tokens=settings.VLLM_MAX_OUTPUT_TOKENS,
            temperature=settings.VLLM_TEMPERATURE
        )
        if not answer_response:
            logger.error("LLM answer generation returned None content. Setting fallback response.")
            answer_response = "I am unable to provide an answer at this time."
            llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + "LLM returned empty answer."
        companion_response_content = answer_response.strip() if isinstance(answer_response, str) else answer_response
        logger.info(f"Generated RAG response using {settings.VLLM_MODEL} for intent '{intent}' (dynamic context).")
        context_snapshot["query_handler_outcome"] = {
            "response_generated": bool(companion_response_content.strip()), 
            "final_error_state": llm_call_error_updated,
            "intent_processed": intent
        }
        return companion_response_content, None, llm_call_error_updated
    except DocumentProcessingError as service_exc:
        error_detail = f"Service error during RAG/QUERY handling ({intent}): {service_exc.message}"
        logger.error(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        companion_response_content = f"I encountered a service issue while trying to find an answer: {service_exc.message}"
    except DatabaseOperationError as db_exc:
        error_detail = f"Database error during RAG/QUERY handling ({intent}): {str(db_exc)}"
        logger.error(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        companion_response_content = "I had trouble accessing necessary data to answer your query."
    except Exception as e_unexp:
        error_detail = f"Unexpected critical error during RAG/QUERY intent handling ({intent}): {str(e_unexp)}"
        logger.critical(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        companion_response_content = "A critical error prevented me from answering your query."
    context_snapshot["query_handler_outcome"] = {
        "response_generated": bool(companion_response_content.strip()), 
        "final_error_state": llm_call_error_updated,
        "intent_processed": intent
    }
    return companion_response_content, None, llm_call_error_updated # No Codex ID linked for queries as no new memory is created here. 