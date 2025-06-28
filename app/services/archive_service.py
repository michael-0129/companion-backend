"""
Archive Service: Centralized logic for all archiving operations in the Companion system.
Supports:
- Archive mode (set/unset, next N)
- Archive by ID
- Archive by tag
- Archive all except (by tag or ID)
- Extensible for future cloud archiving

All methods are classmethods for easy use. Protocol event logging is stubbed for now.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import UUID
from app import models, schemas
from app.services.memory import (
    create_protocol_event, deactivate_protocol_event, get_active_protocol_event
)
from app.core.logging_config import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

class ArchiveService:
    _archive_mode_count: Optional[int] = None  # For 'next N' logic
    _archive_mode_except_ids: Optional[List[UUID]] = None
    _archive_mode_except_tags: Optional[List[str]] = None

    @classmethod
    def activate_archive_mode(cls, db: Session, count: Optional[int] = None, except_ids: Optional[List[UUID]] = None, except_tags: Optional[List[str]] = None):
        """Activate archive mode. Optionally for next N entries, or with exceptions."""
        cls._archive_mode_count = count
        cls._archive_mode_except_ids = except_ids or []
        cls._archive_mode_except_tags = except_tags or []
        details = {"count": count, "except_ids": [str(i) for i in (except_ids or [])], "except_tags": except_tags or []}
        event_schema = schemas.ProtocolEventCreate(
            event_type="archive_mode", active=True, details=details
        )
        create_protocol_event(db, event_schema)
        logger.info(f"Archive mode activated: count={count}, except_ids={except_ids}, except_tags={except_tags}")

    @classmethod
    def deactivate_archive_mode(cls, db: Session):
        """Deactivate archive mode."""
        active_event = get_active_protocol_event(db, "archive_mode")
        if active_event:
            deactivate_protocol_event(db, active_event.id)
        cls._archive_mode_count = None
        cls._archive_mode_except_ids = None
        cls._archive_mode_except_tags = None
        logger.info("Archive mode deactivated.")

    @classmethod
    def should_archive_on_create(cls, entry_id: Optional[UUID] = None, tags: Optional[List[str]] = None) -> bool:
        """Determine if a new entry should be archived based on current archive mode and exceptions."""
        if cls._archive_mode_count is not None:
            if cls._archive_mode_count > 0:
                if entry_id and cls._archive_mode_except_ids and entry_id in cls._archive_mode_except_ids:
                    return False
                if tags and cls._archive_mode_except_tags and any(tag in tags for tag in cls._archive_mode_except_tags):
                    return False
                cls._archive_mode_count -= 1
                if cls._archive_mode_count == 0:
                    # Auto-deactivate after N
                    cls.deactivate_archive_mode(None)
                return True
        return False

    @classmethod
    def archive_entries_by_ids(cls, db: Session, ids: List[UUID]):
        """Archive specific entries by ID."""
        entries = db.query(models.CodexEntry).filter(models.CodexEntry.id.in_(ids)).all()
        for entry in entries:
            entry.archived = True
            entry.protocol_flags = list(set((entry.protocol_flags or []) + ["archived_by_command"]))
        db.commit()
        # Log protocol event
        details = {"archived_ids": [str(i) for i in ids]}
        event_schema = schemas.ProtocolEventCreate(
            event_type="archive_item", active=False, details=details
        )
        create_protocol_event(db, event_schema)
        logger.info(f"Archived entries by ID: {ids}")

    @classmethod
    def archive_entries_by_tag(cls, db: Session, tag: str):
        """Archive all entries with a given tag."""
        entries = db.query(models.CodexEntry).filter(models.CodexEntry.tags.contains([tag]), models.CodexEntry.archived == False).all()
        for entry in entries:
            entry.archived = True
            entry.protocol_flags = list(set((entry.protocol_flags or []) + ["archived_by_tag"]))
        db.commit()
        details = {"archived_tag": tag}
        event_schema = schemas.ProtocolEventCreate(
            event_type="archive_by_tag", active=False, details=details
        )
        create_protocol_event(db, event_schema)
        logger.info(f"Archived entries by tag: {tag}")

    @classmethod
    def archive_all_except(cls, db: Session, except_ids: Optional[List[UUID]] = None, except_tags: Optional[List[str]] = None):
        """Archive all entries except those with given IDs or tags."""
        query = db.query(models.CodexEntry).filter(models.CodexEntry.archived == False)
        if except_ids:
            query = query.filter(~models.CodexEntry.id.in_(except_ids))
        if except_tags:
            for tag in except_tags:
                query = query.filter(~models.CodexEntry.tags.contains([tag]))
        entries = query.all()
        for entry in entries:
            entry.archived = True
            entry.protocol_flags = list(set((entry.protocol_flags or []) + ["archived_by_except"]))
        db.commit()
        details = {"except_ids": [str(i) for i in (except_ids or [])], "except_tags": except_tags or []}
        event_schema = schemas.ProtocolEventCreate(
            event_type="archive_all_except", active=False, details=details
        )
        create_protocol_event(db, event_schema)
        logger.info(f"Archived all entries except: ids={except_ids}, tags={except_tags}")

    @classmethod
    def archive_to_cloud(cls, db: Session, ids: Optional[List[UUID]] = None):
        """Stub for future: Archive entries to the cloud."""
        # Placeholder for future cloud integration
        logger.info(f"[Stub] Would archive to cloud: ids={ids}")
        pass 