from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app import schemas
from app.db.session import get_db
from app.services import documents
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/upload", response_model=schemas.DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> schemas.DocumentUploadResponse:
    """
    Upload and process a document.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    try:
        result = await documents.process_document_upload(
            db=db,
            file=file,
            filename=file.filename
        )
        return result
    except Exception as e:
        logger.error(f"Error processing document upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[schemas.DocumentOut])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[schemas.DocumentOut]:
    """
    List all documents with pagination.
    """
    try:
        return await documents.list_documents(db=db, skip=skip, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/{document_id}", response_model=schemas.DocumentOut)
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_db)
) -> schemas.DocumentOut:
    """
    Get document by ID.
    """
    document = await documents.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    return document

@router.get("/{document_id}/status")
async def get_document_status(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get document processing status.
    """
    document = await documents.get_document(db=db, document_id=document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found"
        )
    return {
        "document_id": document.id,
        "status": document.status,
        "error_message": document.error_message,
        "processed_at": document.processed_at
    }

@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its associated files.
    """
    try:
        await documents.delete_document(db=db, document_id=document_id)
        return {"status": "success", "message": "Document deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/{document_id}/reprocess", response_model=schemas.DocumentOut)
async def reprocess_document(
    document_id: UUID,
    db: Session = Depends(get_db)
) -> schemas.DocumentOut:
    """
    Reprocess a failed document.
    """
    try:
        document = await documents.reprocess_document(db=db, document_id=document_id)
        return document
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# The DocumentProcessResponse schema is defined in schemas.py 