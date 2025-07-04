"""
Service layer for managing memories (CodexEntry), chat history, protocol events, 
document processing, and interactions with OpenAI for embeddings and content extraction.

This module provides the core business logic for data manipulation and AI-assisted 
processing, intended to be used by API routers and other higher-level services.
Error handling within this module primarily raises custom application exceptions defined
in `app.core.exceptions` to allow for consistent error management by callers.
"""
from sqlalchemy.orm import Session
from sqlalchemy import exc as sa_exc
from app import models, schemas
from app.core.config import settings, TIMEZONE
from app.core.logging_config import get_logger
from app.core.exceptions import (
    DatabaseOperationError,
    DocumentProcessingError
)
from app.utils.security import encrypt_content, decrypt_content
from typing import List, Optional
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from app.utils.embeddings import embed_query, embed_passage

logger = get_logger(__name__)

EMBEDDING_MODEL = settings.EMBEDDING_MODEL

cet_tz = ZoneInfo(TIMEZONE)

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
        print("*"*20)
        print(f"[Relevant memories]: {[
            f"Memory from {m.event_date.strftime('%Y-%m-%d') if m.event_date else 'Unknown date'}: {decrypt_content(m.encrypted_content)}"
            for m in results
        ]}")
        print("*"*20)
        logger.info(f"Semantic search found {len(results)} results for query: '{query_text[:50]}...'")
        return results

    except (DocumentProcessingError) as e:
        logger.error(f"Unexpected error during semantic search DB operations for '{query_text[:50]}...': {e}", exc_info=True)
        db.rollback()
        raise DatabaseOperationError(message=f"Unexpected error during semantic search DB operations: {str(e)}", original_exception=e)


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

async def create_codex_entry(
    db: Session,
    entry: schemas.CodexEntryCreate,
    embedding: Optional[list] = None,
) -> models.CodexEntry:
    """
    Create a new codex entry.
    This is a shared function used by both memory and document services.
    """
    try:
        # Encrypt the content before storing
        encrypted_content = encrypt_content(entry.content)
        
        # Create the codex entry
        codex_entry = models.CodexEntry(
            encrypted_content=encrypted_content,
            type=entry.type,
            tags=entry.tags,
            meta=entry.meta,
            entities=entry.entities,
            protocol_flags=entry.protocol_flags,
            event_date=entry.event_date,
            linked_to=entry.linked_to,
            archived=entry.archived,
            vector=embedding if embedding is not None else None
        )
        
        db.add(codex_entry)
        db.commit()
        db.refresh(codex_entry)
        
        return codex_entry
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating codex entry: {str(e)}", exc_info=True)
        raise DocumentProcessingError(
            message=f"Failed to create codex entry: {str(e)}",
            details={"entry_type": entry.type}
        ) 