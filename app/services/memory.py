"""
Service layer for managing memories (CodexEntry), chat history, protocol events, 
document processing, and interactions with OpenAI for embeddings and content extraction.

This module provides the core business logic for data manipulation and AI-assisted 
processing, intended to be used by API routers and other higher-level services.
Error handling within this module primarily raises custom application exceptions defined
in `app.core.exceptions` to allow for consistent error management by callers.
"""
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, desc, or_, exc as sa_exc
from app import models, schemas
from app.core.config import settings, TIMEZONE
from app.core.logging_config import get_logger
from app.core.exceptions import (
    DatabaseOperationError,
    DocumentProcessingError,
    ConfigurationError
)
from app.utils.security import encrypt_content, decrypt_content
from typing import List, Optional, Dict, Any, Tuple
import os
import uuid
from fastapi import UploadFile, HTTPException
import json
import PyPDF2
import io
from datetime import datetime, timezone, timedelta
import uuid as UUID
from app.services.codex_service import create_codex_entry
from zoneinfo import ZoneInfo
from app.utils.llm_provider import get_llm_provider
from app.utils.embeddings import embed_query, embed_passage

logger = get_logger(__name__)

EMBEDDING_MODEL = settings.EMBEDDING_MODEL

cet_tz = ZoneInfo(TIMEZONE)

# List of file extensions supported for direct text extraction.
SUPPORTED_TEXT_EXTENSIONS = [
    ".txt", ".md", ".log", ".rtf", ".html", ".xml", ".json", ".csv", ".tsv", ".ini", ".cfg",
    ".yaml", ".yml", ".py", ".js", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go",
    ".php", ".rb", ".swift", ".kt", ".kts", ".css", ".scss", ".less", ".sql", ".sh", ".ps1",
    ".org"
]

async def generate_embedding(text: str, is_query: bool = False) -> list:
    """
    Generates an embedding for the given text using the local E5 model.
    Args:
        text: The input text to embed.
        is_query: If True, use query prefixing; else, use passage prefixing.
    Returns:
        A list of floats representing the embedding vector.
    Raises:
        DocumentProcessingError: If the input text is empty or invalid.
    """
    if not text or not isinstance(text, str) or not text.strip():
        raise DocumentProcessingError("Input text for embedding cannot be empty or non-string.")
    try:
        if is_query:
            embedding = embed_query(text)
        else:
            embedding = embed_passage(text)
        if not embedding or not isinstance(embedding, list):
            raise DocumentProcessingError("Embedding generation failed: empty or invalid embedding returned.")
        return embedding
    except Exception as e:
        logger.error(f"Unexpected error generating embedding (local E5): {e}", exc_info=True)
        raise DocumentProcessingError(f"Failed to generate embedding: {str(e)}")

async def extract_entities_with_llm(text_content: str, model: str = None) -> List[Dict[str, str]]:
    """Extracts key entities from text content using an LLM.

    Args:
        text_content: The text to analyze for entities.
        model: The LLM model to use for entity extraction. Defaults to settings.VLLM_MODEL.

    Returns:
        A list of dictionaries, where each dictionary represents an entity 
        with "text" and "type" keys. Returns an empty list if no entities are found
        or if the input text is empty.

    Raises:
        DocumentProcessingError: If the LLM response is not valid JSON or has an unexpected structure.
    """
    model = model or settings.VLLM_MODEL
    if not model:
        logger.warning("LLM model not specified. Calls requiring LLM will fail.")
        raise DocumentProcessingError("memory", message="LLM model not specified. Calls requiring LLM will fail.")
    if not text_content or not text_content.strip():
        logger.debug("Skipping entity extraction for empty or whitespace-only text_content.")
        return [] # Valid: empty input yields empty list of entities.
        
    prompt_template = """Extract key entities from the following text.
    Entities can include people, places, organizations, dates, events, concepts, project names, or other significant terms.
    For each extracted entity, provide its text and a suggested type (e.g., PERSON, LOCATION, ORGANIZATION, DATE, EVENT, CONCEPT, PROJECT_NAME, KEY_TERM).
    If the text is very short or contains no clear entities, return an empty list.
    
    Respond ONLY with a single, minified, valid JSON list of objects. Each object must have "text" and "type" keys.
    Example: [{{"text": "Paris", "type": "LOCATION"}}, {{"text": "Project Alpha", "type": "PROJECT_NAME"}}]
    If no entities are found, respond with an empty JSON list: [].
    
    Text to analyze:
    "{text_to_analyze}"
    """
    
    messages = [{"role": "system", "content": prompt_template.format(text_to_analyze=text_content.strip())}]
    response_content_for_logging = "N/A"

    try:
        llm = get_llm_provider()
        response = await llm.generate(
            messages=messages,
            max_tokens=256,
            temperature=settings.VLLM_TEMPERATURE,
            model=model
        )
        response_content = response if isinstance(response, str) else getattr(response, 'content', None)
        response_content_for_logging = response_content[:200] if response_content else "None"

        if response_content:
            data = json.loads(response_content)
            # LLM can return a list directly or a dict like {"entities": []}
            if isinstance(data, list):
                # Validate structure of each item in the list
                if all(isinstance(item, dict) and "text" in item and "type" in item for item in data):
                    return data
                else:
                    logger.warning(f"LLM entity extraction returned a list with invalid items: {response_content_for_logging}")
                    raise DocumentProcessingError("memory", message="LLM entity extraction returned list with invalid items.", details={"model": model, "response_snippet": response_content_for_logging})
            elif isinstance(data, dict) and "entities" in data and isinstance(data["entities"], list):
                 # Wrapped in an "entities" key, validate items in that list
                if all(isinstance(item, dict) and "text" in item and "type" in item for item in data["entities"]):
                    return data["entities"]
                else:
                    logger.warning(f"LLM entity extraction returned dict with invalid 'entities' list: {response_content_for_logging}")
                    raise DocumentProcessingError("memory", message="LLM entity extraction returned dict with invalid 'entities' list.", details={"model": model, "response_snippet": response_content_for_logging})
            elif isinstance(data, dict) and "text" in data and "type" in data:
                # LLM returned a single entity as a dict; wrap in a list
                logger.info(f"LLM entity extraction returned a single entity dict, wrapping in list: {response_content_for_logging}")
                return [data]
            else:
                logger.warning(f"LLM entity extraction returned unexpected JSON structure: {response_content_for_logging}")
                raise DocumentProcessingError("memory", message="LLM entity extraction returned unexpected JSON structure.", details={"model": model, "response_snippet": response_content_for_logging})
        # If LLM returns empty (but valid JSON like []) or None content, it might be intentional (no entities found).
        logger.debug(f"LLM entity extraction returned empty or None content for text: '{text_content[:100]}...'. Returning empty list.")
        return [] 
    except Exception as e:
        logger.error(f"Unexpected error in extract_entities_with_llm (model: {model}). Error: {e}. Snippet: {response_content_for_logging}", exc_info=True)
        raise DocumentProcessingError("memory", message=f"Unexpected error during LLM entity extraction: {str(e)}", details={"model": model})

async def get_codex_entry(
    db: Session,
    entry_id: uuid.UUID,
) -> Optional[models.CodexEntry]:
    """Get a codex entry by ID."""
    entry = db.query(models.CodexEntry).filter(models.CodexEntry.id == entry_id).first()
    if entry and entry.encrypted_content:
        # Decrypt content for output
        entry.content = decrypt_content(entry.encrypted_content)
    return entry

async def update_codex_entry(
    db: Session,
    entry_id: uuid.UUID,
    update_data: schemas.CodexEntryUpdate,
) -> Optional[models.CodexEntry]:
    """Update a codex entry."""
    entry = await get_codex_entry(db, entry_id)
    if not entry:
        return None

    try:
        update_dict = update_data.dict(exclude_unset=True)
        
        # Handle content encryption if it's being updated
        if 'content' in update_dict:
            update_dict['encrypted_content'] = encrypt_content(update_dict.pop('content'))
        
        for field, value in update_dict.items():
            setattr(entry, field, value)

        db.commit()
        db.refresh(entry)
        return entry
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating codex entry: {str(e)}", exc_info=True)
        raise DatabaseOperationError(message=str(e))

async def delete_codex_entry(
    db: Session,
    entry_id: uuid.UUID,
) -> bool:
    """Delete a codex entry."""
    entry = await get_codex_entry(db, entry_id)
    if not entry:
        return False

    try:
        db.delete(entry)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting codex entry: {str(e)}", exc_info=True)
        raise DatabaseOperationError(message=str(e))

async def search_codex_entries(
    db: Session,
    params: schemas.CodexSearchParams,
) -> List[models.CodexEntry]:
    """
    Search codex entries with filtering and semantic search.
    """
    try:
        # Start with base query
        query = db.query(models.CodexEntry)

        # Apply filters
        if not params.include_archived:
            query = query.filter(models.CodexEntry.archived == False)

        if params.types:
            query = query.filter(models.CodexEntry.type.in_(params.types))

        if params.tags:
            for tag in params.tags:
                query = query.filter(models.CodexEntry.tags.contains([tag]))

        if params.start_date:
            query = query.filter(models.CodexEntry.created_at >= params.start_date)

        if params.end_date:
            query = query.filter(models.CodexEntry.created_at <= params.end_date)

        # Get embedding for semantic search
        if params.query:
            query_embedding = await generate_embedding(params.query)
            # Order by cosine similarity
            query = query.order_by(
                models.CodexEntry.vector.cosine_distance(query_embedding)
            )

        # Apply pagination
        query = query.offset(params.skip).limit(params.limit)

        # Execute query and decrypt results
        entries = query.all()
        for entry in entries:
            entry.content = decrypt_content(entry.encrypted_content)

        return entries

    except Exception as e:
        logger.error(f"Error searching codex entries: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search codex entries: {str(e)}")

async def get_related_entries(
    db: Session,
    entry_id: uuid.UUID,
    limit: int = 5,
) -> List[models.CodexEntry]:
    """
    Get semantically related entries for a given entry.
    """
    entry = await get_codex_entry(db, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    try:
        # Get related entries by vector similarity
        query = db.query(models.CodexEntry).filter(
            and_(
                models.CodexEntry.id != entry_id,
                models.CodexEntry.archived == False
            )
        )

        # Order by cosine similarity
        query = query.order_by(
            models.CodexEntry.vector.cosine_distance(entry.vector)
        )

        # Get results and decrypt
        entries = query.limit(limit).all()
        for entry in entries:
            entry.content = decrypt_content(entry.encrypted_content)

        return entries

    except Exception as e:
        logger.error(f"Error getting related entries for {entry_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get related entries: {str(e)}")

async def get_entry_by_type(
    db: Session,
    entry_type: str,
    skip: int = 0,
    limit: int = 20,
) -> List[models.CodexEntry]:
    """
    Get entries of a specific type.
    """
    try:
        entries = db.query(models.CodexEntry).filter(
            and_(
                models.CodexEntry.type == entry_type,
                models.CodexEntry.archived == False
            )
        ).order_by(
            desc(models.CodexEntry.created_at)
        ).offset(skip).limit(limit).all()

        # Decrypt content
        for entry in entries:
            entry.content = decrypt_content(entry.encrypted_content)

        return entries

    except Exception as e:
        logger.error(f"Error getting entries of type {entry_type}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get entries by type: {str(e)}")

async def get_entry_by_tag(
    db: Session,
    tag: str,
    skip: int = 0,
    limit: int = 20,
) -> List[models.CodexEntry]:
    """
    Get entries with a specific tag.
    """
    try:
        entries = db.query(models.CodexEntry).filter(
            and_(
                models.CodexEntry.tags.contains([tag]),
                models.CodexEntry.archived == False
            )
        ).order_by(
            desc(models.CodexEntry.created_at)
        ).offset(skip).limit(limit).all()

        # Decrypt content
        for entry in entries:
            entry.content = decrypt_content(entry.encrypted_content)

        return entries

    except Exception as e:
        logger.error(f"Error getting entries with tag {tag}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get entries by tag: {str(e)}")

def create_chat_history(db: Session, entry: schemas.ChatHistoryCreate) -> models.ChatHistory:
    """Creates a new ChatHistory entry in the database.

    Args:
        db: The SQLAlchemy database session.
        entry: Pydantic schema containing data for the new chat history entry.

    Returns:
        The created ChatHistory model instance.

    Raises:
        DatabaseOperationError: If any database error occurs during creation or commit.
    """
    try:
        db_entry_data = entry.model_dump()
        # Ensure ID and timestamp are set if not provided by the schema (Pydantic defaults should handle this).
        if 'id' not in db_entry_data or not db_entry_data['id']:
            db_entry_data['id'] = uuid.uuid4()
        if 'timestamp' not in db_entry_data or not db_entry_data['timestamp']:
            db_entry_data['timestamp'] = datetime.now(cet_tz)
        
        db_chat_item = models.ChatHistory(**db_entry_data)
        db.add(db_chat_item)
        db.commit()
        db.refresh(db_chat_item)
        logger.info(f"Created ChatHistory entry {db_chat_item.id}.")
        return db_chat_item
    except sa_exc.IntegrityError as e:
        db.rollback()
        logger.error(f"Database IntegrityError creating chat history: {e}", exc_info=True)
        raise DatabaseOperationError(message="Chat history creation failed due to a data conflict.", details={"original_error": str(e)}) from e
    except sa_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database SQLAlchemyError creating chat history: {e}", exc_info=True)
        raise DatabaseOperationError(message="A database error occurred while creating chat history.", details={"original_error": str(e)}) from e

def list_chat_history(db: Session, skip: int = 0, limit: int = 100) -> List[models.ChatHistory]:
    """Lists ChatHistory entries, ordered by timestamp descending.

    Args:
        db: The SQLAlchemy database session.
        skip: Number of entries to skip (for pagination).
        limit: Maximum number of entries to return.

    Returns:
        A list of ChatHistory model instances.

    Raises:
        DatabaseOperationError: If a database error occurs during the query.
    """
    try:
        entries = db.query(models.ChatHistory).order_by(desc(models.ChatHistory.timestamp)).offset(skip).limit(limit).all()
        
        return list(reversed(entries))
    except sa_exc.SQLAlchemyError as e:
        logger.error(f"Database error listing chat history: {e}", exc_info=True)
        raise DatabaseOperationError(message="Failed to list chat history.", details={"original_error": str(e)}) from e

def create_protocol_event(db: Session, event: schemas.ProtocolEventCreate) -> models.ProtocolEvent:
    """Creates a new ProtocolEvent in the database.

    Args:
        db: The SQLAlchemy database session.
        event: Pydantic schema containing data for the new protocol event.

    Returns:
        The created ProtocolEvent model instance.

    Raises:
        DatabaseOperationError: If any database error occurs during creation or commit.
    """
    try:
        db_event_data = event.model_dump()
        if 'id' not in db_event_data or not db_event_data['id']:
            db_event_data['id'] = uuid.uuid4()
        if 'timestamp' not in db_event_data or not db_event_data['timestamp']:
            db_event_data['timestamp'] = datetime.now(cet_tz)
        
        db_protocol_event = models.ProtocolEvent(**db_event_data)
        db.add(db_protocol_event)
        db.commit()
        db.refresh(db_protocol_event)
        logger.info(f"Created ProtocolEvent {db_protocol_event.id} of type '{db_protocol_event.event_type}'. Active: {db_protocol_event.active}")
        return db_protocol_event
    except sa_exc.IntegrityError as e:
        db.rollback()
        logger.error(f"Database IntegrityError creating protocol event: {e}", exc_info=True)
        raise DatabaseOperationError(message="Protocol event creation failed due to a data conflict.", details={"original_error": str(e)}) from e
    except sa_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database SQLAlchemyError creating protocol event: {e}", exc_info=True)
        raise DatabaseOperationError(message="A database error occurred while creating the protocol event.", details={"original_error": str(e)}) from e

def list_protocol_events(
    db: Session, skip: int = 0, limit: int = 100, 
    event_type: Optional[str] = None, active: Optional[bool] = None
) -> List[models.ProtocolEvent]:
    """Lists ProtocolEvents based on filters, ordered by timestamp descending.

    Args:
        db: The SQLAlchemy database session.
        skip: Number of entries to skip.
        limit: Maximum number of entries to return.
        event_type: Filter by specific event type.
        active: Filter by active status (True or False).

    Returns:
        A list of ProtocolEvent model instances.

    Raises:
        DatabaseOperationError: If a database error occurs during the query.
    """
    try:
        query = db.query(models.ProtocolEvent)
        if event_type:
            query = query.filter(models.ProtocolEvent.event_type == event_type)
        if active is not None:
            query = query.filter(models.ProtocolEvent.active == active)
        events = query.order_by(desc(models.ProtocolEvent.timestamp)).offset(skip).limit(limit).all()
        return events
    except sa_exc.SQLAlchemyError as e:
        logger.error(f"Database error listing protocol events: {e}", exc_info=True)
        raise DatabaseOperationError(message="Failed to list protocol events.", details={"original_error": str(e)}) from e

def get_active_protocol_event(db: Session, event_type: str) -> Optional[models.ProtocolEvent]:
    """Retrieves the most recent active ProtocolEvent of a specific type.

    Args:
        db: The SQLAlchemy database session.
        event_type: The type of protocol event to search for.

    Returns:
        The ProtocolEvent model instance if an active one is found, otherwise None.

    Raises:
        DatabaseOperationError: If a database error occurs during the query.
    """
    try:
        return db.query(models.ProtocolEvent).filter(
            models.ProtocolEvent.event_type == event_type,
            models.ProtocolEvent.active == True
        ).order_by(desc(models.ProtocolEvent.timestamp)).first()
    except sa_exc.SQLAlchemyError as e:
        logger.error(f"Database error getting active protocol event for type '{event_type}': {e}", exc_info=True)
        raise DatabaseOperationError(message=f"Failed to get active protocol event for type '{event_type}'.", details={"event_type": event_type, "original_error": str(e)}) from e

def deactivate_protocol_event(db: Session, event_id: uuid.UUID) -> Optional[models.ProtocolEvent]:
    """Deactivates a specific ProtocolEvent by its ID.

    Args:
        db: The SQLAlchemy database session.
        event_id: The UUID of the ProtocolEvent to deactivate.

    Returns:
        The ProtocolEvent model instance if found and deactivated (or was already inactive), 
        otherwise None if not found.

    Raises:
        DatabaseOperationError: If any database error occurs during update or commit.
    """
    try:
        # Fetch the event first.
        db_event = db.query(models.ProtocolEvent).filter(models.ProtocolEvent.id == event_id).first()
        if db_event:
            if db_event.active:
                db_event.active = False
                db.commit() # Commit the change.
                db.refresh(db_event)
                logger.info(f"Deactivated ProtocolEvent {db_event.id} (type: '{db_event.event_type}').")
            else:
                logger.info(f"ProtocolEvent {db_event.id} (type: '{db_event.event_type}') was already inactive.")
            return db_event
        return None # Event not found.
    except sa_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error deactivating protocol event {event_id}: {e}", exc_info=True)
        raise DatabaseOperationError(message=f"Failed to deactivate protocol event {event_id}.", details={"event_id": str(event_id), "original_error": str(e)}) from e
    except Exception as e: # Catch other unexpected errors.
        logger.error(f"Unexpected error deactivating ProtocolEvent {event_id}: {e}", exc_info=True)
        db.rollback() # Attempt rollback.
        raise DatabaseOperationError(message=f"Unexpected error deactivating ProtocolEvent {event_id}: {str(e)}", original_exception=e)

async def semantic_search_codex(
    db: Session, 
    query_text: str, 
    top_k: int = 5,
    start_date_str: Optional[str] = None, 
    end_date_str: Optional[str] = None,   
    entry_type_filter: Optional[List[str]] = None, 
    tag_filter_any: Optional[List[str]] = None, 
    exclude_archived: bool = True
) -> List[models.CodexEntry]:
    """Performs semantic search on CodexEntries using vector similarity.

    Args:
        db: The SQLAlchemy database session.
        query_text: The text to search for.
        top_k: The maximum number of results to return.
        start_date_str: Optional start date filter (YYYY-MM-DD string).
        end_date_str: Optional end date filter (YYYY-MM-DD string).
        entry_type_filter: Optional list of entry types to filter by.
        tag_filter_any: Optional list of tags; entries matching any tag will be included.
        exclude_archived: If True, excludes archived entries.

    Returns:
        A list of CodexEntry model instances matching the search criteria, ordered by relevance.

    Raises:
        DocumentProcessingError: If the query text is invalid for embedding.
        DatabaseOperationError: If a database error occurs during the search query.
    """
    if not query_text or not query_text.strip():
        logger.debug("Semantic search called with empty query text. Returning empty list.")
        return []
    
    query_embedding: List[float]
    try:
        # generate_embedding raises specific errors if it fails.
        query_embedding = await generate_embedding(query_text)
    except (DocumentProcessingError) as e:
        # Log and re-raise to be handled by the API layer.
        logger.error(f"Failed to generate embedding for semantic search query '{query_text[:50]}...': {e}", exc_info=True)
        raise
    except Exception as e: # Catch any other unexpected error specifically from generate_embedding
        logger.error(f"Unexpected error from generate_embedding during semantic search for '{query_text[:50]}...': {e}", exc_info=True)
        # Wrap in DocumentProcessingError as it's related to an OpenAI utility.
        raise DocumentProcessingError(f"Unexpected error generating embedding for search: {str(e)}", original_exception=e)

    # Defensive check, though generate_embedding should raise an error if it can't produce a valid embedding.
    if not query_embedding:
        logger.warning(f"Embedding for semantic search query '{query_text[:50]}...' was empty but no exception raised. Returning no results.")
        return []

    try:
        stmt = db.query(models.CodexEntry)
        
        # Apply date filters if provided and valid.
        if start_date_str:
            try:
                dt_start = datetime.fromisoformat(start_date_str).replace(tzinfo=cet_tz)
                stmt = stmt.filter(models.CodexEntry.event_date >= dt_start)
            except ValueError:
                logger.warning(f"Invalid start_date_str format: '{start_date_str}'. Ignoring filter.")
        if end_date_str:
            try:
                dt_end = datetime.fromisoformat(end_date_str).replace(tzinfo=cet_tz)
                stmt = stmt.filter(models.CodexEntry.event_date <= dt_end)
            except ValueError:
                logger.warning(f"Invalid end_date_str format: '{end_date_str}'. Ignoring filter.")

        if entry_type_filter:
            stmt = stmt.filter(models.CodexEntry.type.in_(entry_type_filter))
        if tag_filter_any:
            stmt = stmt.filter(models.CodexEntry.tags.op('&&')(tag_filter_any))
        if exclude_archived:
            stmt = stmt.filter(models.CodexEntry.archived == False)

        # Order by vector similarity (L2 distance) and limit results
        stmt = stmt.order_by(models.CodexEntry.vector.l2_distance(query_embedding)).limit(top_k)

        # Execute the query
        results = stmt.all()
        logger.info(f"Semantic search found {len(results)} results for query: '{query_text[:50]}...'")
        return results

    except (DocumentProcessingError) as e:
        logger.error(f"Unexpected error during semantic search DB operations for '{query_text[:50]}...': {e}", exc_info=True)
        db.rollback()
        raise DatabaseOperationError(message=f"Unexpected error during semantic search DB operations: {str(e)}", original_exception=e)

def archive_last_n_codex_entries(
    db: Session, 
    count: int, 
    archive_as_type: str, 
    archive_with_tags: Optional[List[str]] = None
) -> List[models.CodexEntry]:
    """Archives the last N non-archived CodexEntries.

    Updates entries to be archived, sets a new type, and adds specified tags.

    Args:
        db: The SQLAlchemy database session.
        count: The number of most recent non-archived entries to archive.
        archive_as_type: The new type to assign to the archived entries.
        archive_with_tags: Optional list of tags to add to the archived entries.

    Returns:
        A list of the updated (archived) CodexEntry model instances.
        Returns an empty list if count is not positive or no entries are found.

    Raises:
        DatabaseOperationError: If any database error occurs during query or commit.
    """
    if count <= 0:
        logger.info("archive_last_n_codex_entries called with count <= 0. No action taken.")
        return []

    try:
        # Fetch the last N non-archived entries.
        entries_to_archive = db.query(models.CodexEntry)\
            .filter(models.CodexEntry.archived == False)\
            .order_by(desc(models.CodexEntry.timestamp))\
            .limit(count)\
            .all()

        if not entries_to_archive:
            logger.info("No non-archived codex entries found to archive.")
            return []

        updated_entry_ids = []
        for entry in entries_to_archive:
            original_type = entry.type # Preserve original type in meta if needed.
            entry.archived = True
            entry.type = archive_as_type 
            
            new_tags = set(entry.tags or []) # Start with existing tags.
            if archive_with_tags:
                new_tags.update(archive_with_tags)
            new_tags.add("auto_archived_by_command") # Add a tag indicating archival method.
            entry.tags = list(new_tags)
            
            # Update meta with archival details.
            current_meta = entry.meta or {}
            current_meta["archival_details"] = {
                "archived_at": datetime.now(cet_tz).isoformat(),
                "original_type": original_type, 
                "archived_by_command": "archive_last_n_inputs", # Specific command/reason
                "archive_as_type": archive_as_type,
                "archive_with_tags_applied": archive_with_tags or []
            }
            entry.meta = current_meta # SQLAlchemy tracks changes to mutable JSONB fields.
            updated_entry_ids.append(entry.id)
        
        db.commit() # Commit all changes for the found entries.
        
        # Fetch the updated entries by ID to return their refreshed state from the DB.
        # This ensures the returned objects reflect the committed changes.
        refreshed_entries = db.query(models.CodexEntry).filter(models.CodexEntry.id.in_(updated_entry_ids)).all()
        logger.info(f"Archived {len(refreshed_entries)} codex entries as type '{archive_as_type}'.")
        return refreshed_entries
    except sa_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error archiving last {count} entries: {e}", exc_info=True)
        raise DatabaseOperationError(message=f"Failed to archive last {count} entries.", details={"count": count, "original_error": str(e)}) from e
    except Exception as e: # Catch-all for other unexpected errors.
        logger.error(f"Unexpected error archiving last {count} entries: {e}", exc_info=True)
        db.rollback()
        raise DatabaseOperationError(message=f"Unexpected error archiving entries: {str(e)}", original_exception=e)


def list_codex_entries(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    type_filter: Optional[str] = None,
    archived: Optional[bool] = None
) -> List[models.CodexEntry]:
    """
    List codex entries with optional filtering.
    
    Args:
        db: Database session
        skip: Number of entries to skip
        limit: Maximum number of entries to return
        type_filter: Optional filter by entry type
        archived: Optional filter by archived status
    Order by event_date descending, fallback to created_at descending if event_date is null.
    """
    query = db.query(models.CodexEntry)
    
    if type_filter:
        query = query.filter(models.CodexEntry.type == type_filter)
    if archived is not None:
        query = query.filter(models.CodexEntry.archived == archived)
    # Order by event_date desc, fallback to created_at desc
    query = query.order_by(
        models.CodexEntry.event_date.desc().nullslast(),
        models.CodexEntry.created_at.desc()
    )
    entries = query.offset(skip).limit(limit).all()
    # Decrypt content for each entry
    for entry in entries:
        if entry.encrypted_content:
            entry.content = decrypt_content(entry.encrypted_content)
    return entries