"""
Relational State Management for Companion

Tracks active archetypes (roles) and closed relational fields, using CodexEntry for persistence.
Provides API for handlers/services to check/set relational state.
All state changes and accesses are logged for auditability and symbolic continuity.
"""
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import cast, String
from datetime import datetime
from app.models import CodexEntry
from app.services.memory import create_codex_entry, get_codex_entry
from app.schemas import CodexEntryCreate
from uuid import UUID
from app.core.config import TIMEZONE
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Constants for CodexEntry type and tags
RESTATE_TYPE = "relational_state"
ACTIVE_ARCHETYPE_KEY = "active_archetype"
CLOSED_FIELDS_KEY = "closed_fields"

cet_tz = ZoneInfo(TIMEZONE)


async def set_active_archetype(db: Session, archetype: str, user: str = "Michael") -> UUID:
    """
    Async. Set the current active archetype (role) for the user. Persists as a CodexEntry.
    All timestamps and event_dates use cet_tz.
    """
    logger.info(f"Setting active archetype to '{archetype}' for user '{user}'.")
    entry = CodexEntryCreate(
        content=f"Active archetype set to: {archetype}",
        type=RESTATE_TYPE,
        tags=[ACTIVE_ARCHETYPE_KEY],
        entities=[{"text": archetype, "type": "ARCHETYPE"}],
        meta={"user": user, "archetype": archetype, "timestamp": datetime.now(cet_tz).isoformat()},
        event_date=datetime.now(cet_tz).date(),
        archived=False,
        protocol_flags=[]
    )
    codex = await create_codex_entry(db, entry, embedding=None)  # No embedding needed for state
    logger.info(f"Active archetype CodexEntry created with ID: {codex.id}")
    return codex.id


async def close_relational_field(db: Session, field: str, user: str = "Michael") -> UUID:
    """
    Async. Mark a relational field (e.g., a person or vector) as closed/sealed. Persists as a CodexEntry.
    All timestamps and event_dates use cet_tz.
    """
    logger.info(f"[SESSION] close_relational_field session id: {id(db)}")
    logger.info(f"Closing relational field '{field}' for user '{user}'.")
    entry = CodexEntryCreate(
        content=f"Relational field closed: {field}",
        type=RESTATE_TYPE,
        tags=[CLOSED_FIELDS_KEY],
        entities=[{"text": field, "type": "RELATIONAL_FIELD"}],
        meta={"user": user, "field": field, "closed": True, "timestamp": datetime.now(cet_tz).isoformat()},
        event_date=datetime.now(cet_tz).date(),
        archived=False,
        protocol_flags=[]
    )
    codex = await create_codex_entry(db, entry, embedding=None)
    logger.info(f"Closed field CodexEntry created with ID: {codex.id}")
    logger.info(f"[DEBUG] Closed field CodexEntry details: id={codex.id}, tags={codex.tags}, meta={codex.meta}, created_at={getattr(codex, 'created_at', None)}, type={codex.type}, entities={codex.entities}")
    # Direct query for diagnostic: is the entry visible in this session?
    from app.models import CodexEntry as ModelCodexEntry
    found = db.query(ModelCodexEntry).filter(
        ModelCodexEntry.id == codex.id
    ).first()
    logger.info(f"[SESSION] After insert, direct query for id={codex.id} found: {found is not None}, tags={getattr(found, 'tags', None)}, meta={getattr(found, 'meta', None)}")
    return codex.id


async def reopen_relational_field(db: Session, field: str, user: str = "Michael") -> UUID:
    """
    Async. Reopen a previously closed relational field. Persists as a CodexEntry.
    All timestamps and event_dates use cet_tz.
    """
    logger.info(f"Reopening relational field '{field}' for user '{user}'.")
    entry = CodexEntryCreate(
        content=f"Relational field reopened: {field}",
        type=RESTATE_TYPE,
        tags=[CLOSED_FIELDS_KEY],
        entities=[{"text": field, "type": "RELATIONAL_FIELD"}],
        meta={"user": user, "field": field, "closed": False, "timestamp": datetime.now(cet_tz).isoformat()},
        event_date=datetime.now(cet_tz).date(),
        archived=False,
        protocol_flags=[]
    )
    codex = await create_codex_entry(db, entry, embedding=None)
    logger.info(f"Reopened field CodexEntry created with ID: {codex.id}")
    return codex.id


def get_current_active_archetype(db: Session, user: str = "Michael") -> Optional[str]:
    """
    Retrieve the most recent active archetype for the user.
    """
    logger.info(f"Retrieving current active archetype for user '{user}'.")
    entry = db.query(CodexEntry).filter(
        CodexEntry.type == RESTATE_TYPE,
        CodexEntry.tags.any(ACTIVE_ARCHETYPE_KEY),
        cast(CodexEntry.meta["user"], String) == user
    ).order_by(CodexEntry.created_at.desc()).first()
    if entry:
        archetype = entry.meta.get("archetype")
        logger.info(f"Current active archetype: {archetype}")
        return archetype
    logger.info("No active archetype found.")
    return None


def get_closed_fields(db: Session, user: str = "Michael") -> List[str]:
    """
    Retrieve all currently closed relational fields for the user.
    """
    logger.info(f"[SESSION] get_closed_fields session id: {id(db)}")
    logger.info(f"Retrieving closed relational fields for user '{user}'.")
    entries = db.query(CodexEntry).filter(
        CodexEntry.type == RESTATE_TYPE,
        CodexEntry.tags.any(CLOSED_FIELDS_KEY),  # ✅ correct for ARRAY(String)
        # CodexEntry.meta["user"].astext == user   # ✅ JSON access in PostgreSQL
    ).order_by(CodexEntry.created_at.desc()).all()
    logger.info(f"[DEBUG] Raw closed field entries found: [" + ", ".join(f"id={e.id}, field={e.meta.get('field')}, closed={e.meta.get('closed')}, created_at={e.created_at}" for e in entries) + "]")
    closed = set()
    reopened = set()
    for entry in entries:
        field = entry.meta.get("field")
        if field is not None:
            norm_field = field.strip().lower()
            if entry.meta.get("closed"):
                closed.add(norm_field)
            else:
                reopened.add(norm_field)
    # Only return fields that are closed and not reopened
    result = list(closed - reopened)
    logger.info(f"Closed fields: {result}")
    return result


def is_field_closed(db: Session, field: str, user: str = "Michael") -> bool:
    """
    Check if a relational field is currently closed for the user.
    """
    closed_fields = get_closed_fields(db, user)
    norm_field = field.strip().lower()
    is_closed = norm_field in closed_fields
    logger.info(f"Field '{field}' closed: {is_closed}")
    return is_closed
