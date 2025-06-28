from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func
from app import schemas
from app.db.session import get_db
from app.core.config import settings
from app.services import memory
from app.utils.security import decrypt_content
from uuid import UUID
from typing import List, Optional, Any, Dict
import json # Added for potential use in new endpoints or schema handling
from app.services.codex_service import create_codex_entry
from app.models import CodexEntry as models

router = APIRouter()

@router.post("/entries", response_model=schemas.CodexEntryOut)
async def create_entry(
    entry: schemas.CodexEntryCreate,
    db: Session = Depends(get_db)
) -> schemas.CodexEntryOut:
    """Create a new codex entry."""
    try:
        db_entry = await create_codex_entry(db=db, entry=entry)
        # Convert to output schema with decrypted content
        return schemas.CodexEntryOut(
            id=db_entry.id,
            content=entry.content,  # Use original content
            tags=db_entry.tags,
            entities=db_entry.entities,
            meta=db_entry.meta,
            archived=db_entry.archived,
            created_at=db_entry.created_at,
            updated_at=db_entry.updated_at,
            type=db_entry.type,
            linked_to=db_entry.linked_to,
            protocol_flags=db_entry.protocol_flags,
            event_date=db_entry.event_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entries/{entry_id}", response_model=schemas.CodexEntryOut)
async def get_entry(
    entry_id: UUID,
    db: Session = Depends(get_db)
) -> schemas.CodexEntryOut:
    """Get a codex entry by ID."""
    entry = await memory.get_codex_entry(db=db, entry_id=entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    
    # Convert to output schema with decrypted content
    return schemas.CodexEntryOut(
        id=entry.id,
        content=decrypt_content(entry.encrypted_content),
        tags=entry.tags,
        entities=entry.entities,
        meta=entry.meta,
        archived=entry.archived,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        type=entry.type,
        linked_to=entry.linked_to,
        protocol_flags=entry.protocol_flags,
        event_date=entry.event_date
    )

@router.get("/entries", response_model=List[schemas.CodexEntryOut])
async def list_entries(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    type_filter: Optional[str] = None,
    archived: Optional[bool] = None,
    db: Session = Depends(get_db)
) -> List[schemas.CodexEntryOut]:
    """List codex entries with optional filtering."""
    try:
        entries = memory.list_codex_entries(
            db=db,
            skip=skip,
            limit=limit,
            type_filter=type_filter,
            archived=archived
        )
        
        # Convert to output schema
        return [
            schemas.CodexEntryOut(
                id=entry.id,
                content=entry.content,  # Already decrypted in list_codex_entries
                tags=entry.tags,
                entities=entry.entities,
                meta=entry.meta,
                archived=entry.archived,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                type=entry.type,
                linked_to=entry.linked_to,
                protocol_flags=entry.protocol_flags,
                event_date=entry.event_date
            )
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/entries/{entry_id}", response_model=schemas.CodexEntryOut)
async def update_entry(entry_id: UUID, entry_update: schemas.CodexEntryUpdate, db: Session = Depends(get_db)):
    db_entry = await memory.update_codex_entry(db, entry_id, entry_update)
    if not db_entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    content_to_return = "Error: Content could not be prepared for response."
    if entry_update.content is not None: 
        content_to_return = entry_update.content
    else:
        content_to_return = decrypt_content(db_entry.encrypted_content)

    return schemas.CodexEntryOut(
        id=db_entry.id,
        content=content_to_return,
        tags=db_entry.tags,
        entities=db_entry.entities,
        meta=db_entry.meta,
        archived=db_entry.archived,
        created_at=db_entry.created_at,
        updated_at=db_entry.updated_at,
        type=db_entry.type,
        linked_to=db_entry.linked_to,
        protocol_flags=db_entry.protocol_flags
    )

@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: UUID, db: Session = Depends(get_db)):
    success = await memory.delete_codex_entry(db, entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"ok": True}

@router.get("/semantic_search", response_model=List[schemas.CodexEntryOut])
async def semantic_search(query: str = Query(...), top_k: int = Query(default=settings.MAX_SEMANTIC_SEARCH_RESULTS, ge=1, le=50), db: Session = Depends(get_db)):
    results = await memory.semantic_search_codex(db, query, top_k)
    output = []
    for row in results:
        db_entry = row[0] if hasattr(row, '__getitem__') else row
        content = decrypt_content(db_entry.encrypted_content)
        output.append(schemas.CodexEntryOut(
            id=db_entry.id,
            content=content,
            tags=db_entry.tags,
            entities=db_entry.entities,
            meta=db_entry.meta,
            archived=db_entry.archived,
            created_at=db_entry.created_at,
            updated_at=db_entry.updated_at,
            type=db_entry.type,
            linked_to=db_entry.linked_to,
            protocol_flags=db_entry.protocol_flags
        ))
    return output 

@router.get("/entries/by-document/{document_id}", response_model=List[schemas.CodexEntryOut])
async def list_entries_by_document(
    document_id: UUID,
    db: Session = Depends(get_db)
) -> List[schemas.CodexEntryOut]:
    """List codex entries (memories) for a given document by source_document_id in meta."""
    try:
        # Detect DB dialect
        dialect = str(db.bind.dialect.name)
        entries = []
        if dialect == 'sqlite':
            # SQLite: use json_extract
            entries = db.query(models).filter(
                func.json_extract(models.meta, '$.source_document_id') == str(document_id)
            ).order_by(models.created_at.desc()).all()
        else:
            # For all other DBs, fallback to Python-side filtering
            all_entries = db.query(models).all()
            entries = [e for e in all_entries if e.meta and e.meta.get("source_document_id") == str(document_id)]
        # Decrypt content for each entry if needed
        result = []
        for entry in entries:
            if hasattr(entry, 'encrypted_content'):
                from app.utils.security import decrypt_content
                entry.content = decrypt_content(entry.encrypted_content)
            result.append(schemas.CodexEntryOut(
                id=entry.id,
                content=entry.content,
                tags=entry.tags,
                entities=entry.entities,
                meta=entry.meta,
                archived=entry.archived,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                type=entry.type,
                linked_to=entry.linked_to,
                protocol_flags=entry.protocol_flags,
                event_date=entry.event_date
            ))
        return result
    except Exception as e:
        import traceback
        print('Error in by-document endpoint:', traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 