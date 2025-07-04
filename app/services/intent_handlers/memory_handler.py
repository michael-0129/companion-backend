"""
Handler for processing intents related to creating and storing memories (CodexEntry items).

This module is responsible for taking classified memory data, generating necessary
embeddings, and interfacing with the memory service layer to persist these memories.
It handles both regular memory creation from user queries and forced archival of inputs.
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Tuple, List
from uuid import UUID
from datetime import datetime # timezone not explicitly used but good for awareness
from fastapi import HTTPException # Current error handling catches this, though services aim to raise custom exceptions.
import re
import dateparser
from zoneinfo import ZoneInfo
from app.core.config import TIMEZONE

from app import schemas # models are used by services, schemas by handlers for data creation.
from app.core.logging_config import get_logger
from app.services.memory import (
    generate_embedding,
    # Import custom exceptions that might be raised by the services
    DatabaseOperationError,
    DocumentProcessingError
)
from app.services.memory import create_codex_entry
from app.services.archive_service import ArchiveService
from .intent_registry import intent_handler_registry
from app.services import relational_state  # Import the new relational state module

logger = get_logger(__name__)

cet_tz = ZoneInfo(TIMEZONE)

@intent_handler_registry.register("MEMORY")
@intent_handler_registry.register("FORCED_ARCHIVE_MEMORY")
async def handle_memory_intent(
    db: Session,
    user_query: str, 
    classification_data: Dict[str, Any],
    context_snapshot: Dict[str, Any], 
    silence_effectively_active: bool,
    current_llm_call_error: Optional[str] 
) -> Tuple[str, Optional[UUID], Optional[str]]:
    """
    Handles "MEMORY" and "FORCED_ARCHIVE_MEMORY" intents by creating a CodexEntry.
    Now supports both legacy and new multi-task formats (parameters dict).

    This function processes data extracted during intent classification (or provided
    directly for forced archival), generates an embedding for the content, and then
    creates a new CodexEntry in the database.

    Args:
        db: The SQLAlchemy database session.
        user_query: The original user query. Used as fallback content if LLM fails to extract specific memory content.
        classification_data: A dictionary containing the results of intent classification, 
                             including "memory_content", "memory_type", "memory_tags", "event_date", etc.
        context_snapshot: A dictionary for storing metadata and logs about the current interaction.
        silence_effectively_active: Boolean indicating if silence mode is currently active.
        current_llm_call_error: An optional string containing error messages from prior steps (e.g., classification).

    Returns:
        A tuple containing:
        - companion_response_content (str): The response text to be sent to the user.
        - linked_codex_id (Optional[UUID]): The UUID of the created CodexEntry if successful, otherwise None.
        - llm_call_error_updated (Optional[str]): The updated string of accumulated error messages.
    
    Raises:
        This function aims to catch exceptions from called services (like `generate_embedding`, `create_codex_entry`)
        and convert them into appropriate user responses and error logs. Expected exceptions from services include:
        - `DocumentProcessingError`: If content for embedding is invalid.
        - `DatabaseOperationError`: For issues during database interaction when creating the codex entry.
        It also currently has a catch for `HTTPException`, though service layers are moving away from raising this directly.
    """
    linked_codex_id = None  # Ensure always defined for all return paths
    # Always initialize error string at the very top
    llm_call_error_updated = current_llm_call_error

    # Accept parameters dict if present (from orchestrator multi-task), else fallback to classification_data
    parameters = classification_data.get("parameters") if "parameters" in classification_data else classification_data
    intent = parameters.get("intent", classification_data.get("intent", "MEMORY"))
    logger.info(f"Handling intent: '{intent}' in memory_handler for user query: '{user_query[:100]}...'")
    logger.info(f"MEMORY HANDLER DEBUG: parameters={parameters}, user_query='{user_query}'")

    # --- Relational State Enforcement ---
    # Check if any referenced relational field is closed
    referenced_fields = []
    if "extracted_entities" in parameters:
        referenced_fields = [e["text"] for e in parameters["extracted_entities"] if e.get("text")]
    closed_fields = [f.lower().strip() for f in relational_state.get_closed_fields(db)]
    for field in referenced_fields:
        if field.lower().strip() in closed_fields:
            logger.warning(f"[PROTOCOL BLOCK] Attempted to create memory for closed field '{field}'. Blocking creation. User query: '{user_query}'. Context: {parameters}")
            # Always return protocol-aligned message, never generic fallback
            protocol_message = f"The field '{field}' is sealed. No new memory may be recorded for it unless explicitly reopened."
            llm_call_error_updated = (current_llm_call_error + "; " if current_llm_call_error else "") + f"Attempted to create memory for closed field: {field}."
            # Log the protocol block for audit, using cet_tz for timestamp
            context_snapshot["protocol_blocked_memory"] = {
                "field": field,
                "user_query": user_query,
                "parameters": parameters,
                "timestamp": datetime.now(cet_tz).isoformat()
            }
            logger.info(f"[PROTOCOL BLOCK LOGGED] Blocked memory creation for field '{field}'. Context snapshot: {context_snapshot['protocol_blocked_memory']}")
            return protocol_message, None, llm_call_error_updated
    # --- End Relational State Enforcement ---

    # --- Archetype Enforcement ---
    # If this memory is archetype-related, ensure only the current active archetype is referenced
    current_archetype = relational_state.get_current_active_archetype(db)
    if current_archetype and "archetype" in parameters:
        if parameters["archetype"] != current_archetype:
            logger.info(f"Overriding referenced archetype '{parameters['archetype']}' with current active archetype '{current_archetype}'.")
            parameters["archetype"] = current_archetype

    # Fallback for event_date if missing/null and query suggests recency
    event_date = parameters.get("event_date")
    if not event_date or str(event_date).strip().lower() in ("null", "none", ""):
        if re.search(r"\b(today|now|recent|conflict|just|this day|earlier)\b", user_query, re.IGNORECASE):
            today_str = datetime.now(cet_tz).strftime("%Y-%m-%d")
            parameters["event_date"] = today_str
            event_date = today_str
            context_snapshot["event_date_fallback"] = f"event_date set to {today_str} based on user_query context."

    # Extract and parse the event date from parameters.
    event_date_str = parameters.get("event_date")
    event_date_for_memory: datetime
    if event_date_str:
        # Use dateparser to handle relative/natural language dates
        parsed_date = dateparser.parse(str(event_date_str), settings={"TIMEZONE": TIMEZONE, "RETURN_AS_TIMEZONE_AWARE": True})
        if parsed_date:
            # Normalize to midnight unless time is explicitly provided
            event_date_for_memory = parsed_date.date()
        else:
            logger.warning(f"Could not parse event_date '{event_date_str}' with dateparser. Defaulting to current time.")
            event_date_for_memory = datetime.now(cet_tz).date()
    else:
        logger.debug(f"No event_date provided in parameters for intent '{intent}'. Defaulting to current time.")
        event_date_for_memory = datetime.now(cet_tz).date()

    # Extract memory content from parameters.
    mem_content = parameters.get("memory_content")
    # Fallback to user_query if memory_content is not specifically extracted by LLM for a standard MEMORY intent.
    if not mem_content and intent == "MEMORY": 
        mem_content = user_query
        logger.warning(f"Memory content was empty from LLM for '{intent}' intent, using raw user_query as content.")

    # If no content is available even after fallback, cannot create a memory.
    if not mem_content or not mem_content.strip(): 
        logger.error(f"Cannot create memory: content is empty or whitespace for intent '{intent}'.")
        if not silence_effectively_active:
            companion_response_content = "Received, Michael. I tried to record this, but there was no content to save."
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + "Memory content was empty and could not be recorded."
        return companion_response_content, linked_codex_id, llm_call_error_updated

    # Determine memory type and tags based on intent and parameters.
    default_type = "archived_item_protocol" if intent == "FORCED_ARCHIVE_MEMORY" else "observation_agent"
    mem_type = parameters.get("memory_type") or default_type
    
    mem_tags = parameters.get("memory_tags", [])
    # Ensure FORCED_ARCHIVE_MEMORY always includes its specific tag if not already present from parameters.
    if intent == "FORCED_ARCHIVE_MEMORY" and "auto_archived_by_protocol" not in mem_tags:
        mem_tags.append("auto_archived_by_protocol")
        
    # Extract and validate entities.
    mem_entities_raw = parameters.get("extracted_entities", [])
    validated_entities: List[Dict[str,str]] = [] # Ensure it's initialized as list of dicts
    if isinstance(mem_entities_raw, list):
        for entity_item in mem_entities_raw:
            if isinstance(entity_item, dict) and "text" in entity_item and "type" in entity_item:
                validated_entities.append(entity_item)
            elif isinstance(entity_item, str): # Handle if LLM sometimes returns a list of strings for entities.
                logger.warning(f"Entity item for memory intent '{intent}' was a string: '{entity_item}'. Converting to standard entity format.")
                validated_entities.append({"text": entity_item, "type": "unknown_llm_provided_string"})
            else:
                logger.warning(f"Skipping invalid entity item during memory creation: {entity_item}")
    else:
        logger.warning(f"Extracted entities for memory intent '{intent}' were not a list: {mem_entities_raw}. Defaulting to empty list.")

    # Prepare metadata for the CodexEntry.
    codex_meta = {"source": "agent_interaction", "intent_handled": intent, "user_query_snippet": user_query[:250]}
    # For regular MEMORY intents, store more detailed classification data if available.
    if intent == "MEMORY" and "classification_details" in parameters:
        codex_meta["classification_details"] = parameters.get("classification_details", parameters) 
    # For FORCED_ARCHIVE_MEMORY, store details about what triggered the archival if available in context_snapshot.
    elif intent == "FORCED_ARCHIVE_MEMORY" and "forced_archival_details" in context_snapshot:
        codex_meta["forced_archival_trigger_details"] = context_snapshot["forced_archival_details"]

    # If there was a classification error before this step (for non-forced memories), log it in meta.
    if llm_call_error_updated and intent != "FORCED_ARCHIVE_MEMORY":
            codex_meta["prior_classification_error_message"] = llm_call_error_updated

    try:
        # Step 1: Generate embedding for the memory content.
        embedding = await generate_embedding(mem_content)  # No await, now sync
        if not embedding:
            logger.error(f"Failed to generate embedding for memory content (intent: {intent}). Cannot save memory. Service returned empty embedding without error.")
            if not silence_effectively_active:
                companion_response_content = "Received, Michael. I tried to record this memory, but an internal error occurred during embedding generation."
            llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + "Embedding generation failed for memory (empty embedding)."
            return companion_response_content, linked_codex_id, llm_call_error_updated

        # Step 2: Prepare data and create the CodexEntry.
        # --- ARCHIVE SERVICE LOGIC ---
        should_archive = ArchiveService.should_archive_on_create(tags=list(set(mem_tags)))
        protocol_flags = [*parameters.get("protocol_flags", [])]
        if should_archive:
            protocol_flags.append("archive_mode")
        codex_entry_data = schemas.CodexEntryCreate(
            content=mem_content, 
            type=mem_type, 
            tags=list(set(mem_tags)),
            entities=validated_entities, 
            meta=codex_meta,
            event_date=event_date_for_memory,
            archived=should_archive,
            protocol_flags=protocol_flags
        )
        # db_codex_entry = await create_codex_entry(db, codex_entry_data, embedding)
        db_codex_entry = await create_codex_entry(db, codex_entry_data, embedding)
        linked_codex_id = db_codex_entry.id
        # Prepare success response for the user if not in silence mode.
        if not silence_effectively_active:
            if intent == "FORCED_ARCHIVE_MEMORY" or should_archive:
                response_type_display = mem_type 
                companion_response_content = f"Received, Michael. By decree, this memory is consigned to the vault: archived as '{response_type_display}' (ID: {db_codex_entry.id})."
            else:
                companion_response_content = "Received, Michael. Memory stored."
        logger.info(f"[MEMORY CREATED] CodexEntry {db_codex_entry.id} of type '{mem_type}' via '{intent}' intent. User query: '{user_query}'. Parameters: {parameters}. Meta: {codex_meta}")
        context_snapshot["memory_creation_details"] = {
            "codex_id": str(db_codex_entry.id), 
            "type": mem_type, 
            "tags": mem_tags, 
            "handler": "memory_handler"
        }

    # Catch specific custom exceptions from the service layer first.
    except DocumentProcessingError as service_exc:
        error_detail = f"Service error during memory creation ({intent}): {str(service_exc)}"
        logger.error(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        if not silence_effectively_active:
            companion_response_content = f"Received, Michael. I tried to record this memory, but a service error occurred: {service_exc.message or str(service_exc)}"
    except DatabaseOperationError as db_exc:
        error_detail = f"Database error during memory creation ({intent}): {str(db_exc)}"
        logger.error(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        if not silence_effectively_active:
            companion_response_content = "Received, Michael. I tried to record this memory, but a database error occurred."
    except HTTPException as http_exc: # Should ideally not be raised by services, but caught if it is.
        error_detail = f"HTTPException during memory creation ({intent}): {http_exc.detail}"
        logger.error(error_detail, exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + f"Failed to save memory due to an API error: {http_exc.detail}"
        if not silence_effectively_active:
            companion_response_content = f"Received, Michael. I tried to record this memory, but an API error occurred: {http_exc.detail}"
    except Exception as e: # Catch-all for any other unexpected errors.
        error_detail = f"Unexpected error during memory creation ({intent}): {str(e)}"
        logger.critical(error_detail, exc_info=True) # Critical for unexpected errors.
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + error_detail
        if not silence_effectively_active:
            companion_response_content = f"Received, Michael. I tried to record this memory, but an unexpected internal error occurred."
            
    return companion_response_content, linked_codex_id, llm_call_error_updated 