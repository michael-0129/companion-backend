"""
Codex service module for handling codex entries.
This module contains shared functionality used by both memory and document services.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from app import models, schemas
from app.core.logging_config import get_logger
from app.core.exceptions import DocumentProcessingError
from app.utils.security import encrypt_content, decrypt_content

logger = get_logger(__name__)

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