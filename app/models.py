from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, ForeignKey, func, Index, Date
from sqlalchemy.dialects.postgresql import UUID, BYTEA, ARRAY
from pgvector.sqlalchemy import VECTOR
import uuid
from sqlalchemy.orm import relationship

from app.db.session import Base

class CodexEntry(Base):
    __tablename__ = "codex_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    encrypted_content = Column(BYTEA, nullable=False)
    tags = Column(ARRAY(String), default=list)
    entities = Column(JSON, default=list)  # Changed from ARRAY(String) to JSON
    meta = Column(JSON, default=dict)
    vector = Column(VECTOR(384), nullable=True)  # pgvector extension, 1536 for default embedding model
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    event_date = Column(Date, nullable=True, index=True)
    archived = Column(Boolean, default=False)
    type = Column(String, default="general")  # e.g., dream, directive, event, doctrinal, etc.
    linked_to = Column(UUID(as_uuid=True), ForeignKey('codex_entries.id'), nullable=True)  # for threading/relational links
    protocol_flags = Column(ARRAY(String), default=list)  # e.g., ["archive_mode", "doctrinal_draft"]
    documents = relationship('Document', backref='codex_entry', primaryjoin="CodexEntry.id==Document.codex_entry_id")

    # Optional: Add a composite index if you often query by type and event_date
    # __table_args__ = (Index('ix_codex_entries_type_event_date', 'type', 'event_date'),)

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_query = Column(String, nullable=False)
    companion_response = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    linked_codex_entry = Column(UUID(as_uuid=True), ForeignKey('codex_entries.id', ondelete='SET NULL'), nullable=True)
    context_snapshot = Column(JSON, default=dict)
    summary = Column(String, nullable=True)  # Concise LLM-generated summary of this turn

class ProtocolEvent(Base):
    __tablename__ = "protocol_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    event_type = Column(String, nullable=False)  # e.g., silence_mode, archive_mode, doctrinal_draft
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    active = Column(Boolean, default=True)

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # e.g., pdf, txt, docx
    file_size = Column(Integer, nullable=False)  # in bytes
    original_file_path = Column(String, nullable=True)  # optional: path to stored file if we keep originals
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, nullable=False, default='processing')  # processing, completed, failed
    error_message = Column(String, nullable=True)
    codex_entry_id = Column(UUID(as_uuid=True), ForeignKey('codex_entries.id', ondelete='CASCADE'), nullable=True)
    doc_metadata = Column(JSON, default=dict)  # flexible field for additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_attempts = Column(Integer, default=0)  # track retry attempts

    __table_args__ = (
        Index('ix_documents_status_upload_date', 'status', 'upload_date'),  # For efficient status-based queries
        Index('ix_documents_codex_entry_id', 'codex_entry_id'),  # For efficient joins with codex entries
    )

    @property
    def metadata_safe(self):
        return self.doc_metadata or {} 