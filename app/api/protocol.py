from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import schemas
from app.db.session import get_db
from app.services import protocol
from uuid import UUID
from typing import List

router = APIRouter()

@router.post("/", response_model=schemas.ProtocolEventOut)
def create_protocol_event(event: schemas.ProtocolEventCreate, db: Session = Depends(get_db)):
    db_event = protocol.create_protocol_event(db, event)
    return schemas.ProtocolEventOut(
        id=db_event.id,
        event_type=db_event.event_type,
        details=db_event.details,
        active=db_event.active,
        timestamp=db_event.timestamp
    )

@router.get("/", response_model=List[schemas.ProtocolEventOut])
def list_protocol_events(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    db_events = protocol.list_protocol_events(db, skip, limit)
    return [
        schemas.ProtocolEventOut(
            id=e.id,
            event_type=e.event_type,
            details=e.details,
            active=e.active,
            timestamp=e.timestamp
        ) for e in db_events
    ]

@router.post("/deactivate/{event_id}")
def deactivate_protocol_event(event_id: UUID, db: Session = Depends(get_db)):
    success = protocol.deactivate_protocol_event(db, event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True} 