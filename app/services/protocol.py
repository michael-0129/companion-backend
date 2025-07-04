from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uuid
from datetime import datetime
import sqlalchemy.exc as sa_exc

from app import models, schemas
from app.core.exceptions import DatabaseOperationError
from app.core.logging_config import get_logger
from zoneinfo import ZoneInfo
from app.core.config import TIMEZONE

logger = get_logger(__name__)
cet_tz = ZoneInfo(TIMEZONE)

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