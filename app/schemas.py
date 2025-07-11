from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, date

class Entity(BaseModel):
    text: str
    type: str

class CodexEntryBase(BaseModel):
    content: str
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    archived: bool = False
    type: str = "general"
    linked_to: Optional[UUID] = None
    protocol_flags: List[str] = Field(default_factory=list)
    event_date: Optional[date] = None

class CodexEntryCreate(CodexEntryBase):
    pass

class CodexEntryUpdate(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None
    archived: Optional[bool] = None
    type: Optional[str] = None
    linked_to: Optional[UUID] = None
    protocol_flags: Optional[List[str]] = None
    event_date: Optional[date] = None

class CodexEntryOut(CodexEntryBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    event_date: Optional[date] = None

    class Config:
        from_attributes = True

class ChatHistoryBase(BaseModel):
    user_query: str
    companion_response: str
    linked_codex_entry: Optional[UUID] = None
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = Field(default=None, description="LLM-generated concise summary of this chat turn.")

class ChatHistoryCreate(ChatHistoryBase):
    pass

class ChatHistoryOut(ChatHistoryBase):
    id: UUID
    timestamp: datetime

    class Config:
        from_attributes = True

class ChatHistoryFilter(BaseModel):
    skip: int = 0
    limit: int = 20
    search_query: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class ProtocolEventBase(BaseModel):
    event_type: str
    details: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True

class ProtocolEventCreate(ProtocolEventBase):
    pass

class ProtocolEventOut(ProtocolEventBase):
    id: UUID
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Agent Interaction Schemas ---
class AgentInteractionRequest(BaseModel):
    user_query: str

# --- Document Processing Schemas ---
class DocumentBase(BaseModel):
    filename: str
    file_type: str
    file_size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    status: Optional[str] = None
    error_message: Optional[str] = None
    codex_entry_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = None
    processed_at: Optional[datetime] = None
    processing_attempts: Optional[int] = None

class DocumentOut(DocumentBase):
    id: UUID
    upload_date: datetime
    status: str
    error_message: Optional[str] = None
    codex_entry_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    processing_attempts: int

    class Config:
        from_attributes = True

class DocumentUploadResponse(BaseModel):
    document_id: UUID
    status: str
    message: str
    codex_entry_id: Optional[UUID] = None

class CodexSearchParams(BaseModel):
    query: str
    types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_archived: bool = False

# Document process response (for backward compatibility)
class DocumentProcessResponse(BaseModel):
    success: bool
    message: str
    codex_entries: List[UUID] = Field(default_factory=list) 

class AuthLoginRequest(BaseModel):
    username: str
    password: str

class AuthLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    message: Optional[str] = None 