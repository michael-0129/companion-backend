from typing import List, Optional
from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app import models, schemas
from app.core.logging_config import get_logger
from app.prompts.system_prompts import SYSTEM_PROMPT_SUMMARIZE_TURN, SYSTEM_PROMPT_SUMMARIZE_CONTEXT
from app.utils.llm_provider import get_llm_provider
import uuid
from zoneinfo import ZoneInfo
from sqlalchemy import exc as sa_exc
from app.core.config import TIMEZONE
from app.core.exceptions import DatabaseOperationError
cet_tz = ZoneInfo(TIMEZONE)

logger = get_logger(__name__)

def create_chat_history(db: Session, entry: schemas.ChatHistoryCreate) -> models.ChatHistory:
    """Creates a new ChatHistory entry in the database.

    Args:
        db: The SQLAlchemy database session.
        entry: Pydantic schema containing data for the new chat history entry.

    Returns:
        The created ChatHistory model instance.

    Raises:
        DatabaseOperationError: If any database error occurs during creation or commit.
    """
    try:
        db_entry_data = entry.model_dump()
        # Ensure ID and timestamp are set if not provided by the schema (Pydantic defaults should handle this).
        if 'id' not in db_entry_data or not db_entry_data['id']:
            db_entry_data['id'] = uuid.uuid4()
        if 'timestamp' not in db_entry_data or not db_entry_data['timestamp']:
            db_entry_data['timestamp'] = datetime.now(cet_tz)
        
        db_chat_item = models.ChatHistory(**db_entry_data)
        db.add(db_chat_item)
        db.commit()
        db.refresh(db_chat_item)
        logger.info(f"Created ChatHistory entry {db_chat_item.id}.")
        return db_chat_item
    except sa_exc.IntegrityError as e:
        db.rollback()
        logger.error(f"Database IntegrityError creating chat history: {e}", exc_info=True)
        raise DatabaseOperationError(message="Chat history creation failed due to a data conflict.", details={"original_error": str(e)}) from e
    except sa_exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database SQLAlchemyError creating chat history: {e}", exc_info=True)
        raise DatabaseOperationError(message="A database error occurred while creating chat history.", details={"original_error": str(e)}) from e


async def get_chat_history(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_query: Optional[str] = None,
) -> List[models.ChatHistory]:
    """
    Retrieve chat history with optional filtering and pagination.
    """
    query = db.query(models.ChatHistory)
    
    # Apply date filters if provided
    if start_date:
        query = query.filter(models.ChatHistory.timestamp >= start_date)
    if end_date:
        query = query.filter(models.ChatHistory.timestamp <= end_date)
    
    # Apply search filter if provided
    if search_query:
        search_filter = or_(
            models.ChatHistory.user_query.ilike(f"%{search_query}%"),
            models.ChatHistory.companion_response.ilike(f"%{search_query}%")
        )
        query = query.filter(search_filter)
    
    # Apply pagination and return results
    results = query.order_by(models.ChatHistory.timestamp.desc()).offset(skip).limit(limit).all()
    return list(reversed(results))
async def get_chat_entry(db: Session, chat_id: UUID) -> Optional[models.ChatHistory]:
    """
    Retrieve a specific chat history entry by ID.
    """
    return db.query(models.ChatHistory).filter(models.ChatHistory.id == chat_id).first()

async def delete_chat_entry(db: Session, chat_id: UUID) -> bool:
    """
    Delete a specific chat history entry.
    Returns True if successful, False if entry not found.
    """
    entry = await get_chat_entry(db, chat_id)
    if not entry:
        return False
    
    db.delete(entry)
    db.commit()
    return True

async def generate_and_save_chat_summary(db, chat_history_instance):
    """
    Generates a concise summary for a chat turn using the LLM and saves it to the summary field.
    Args:
        db: SQLAlchemy session
        chat_history_instance: models.ChatHistory instance (must be committed)
    """
    if not chat_history_instance or not chat_history_instance.id:
        logger.error("No valid chat history instance provided for summary generation.")
        return
    try:
        prompt = SYSTEM_PROMPT_SUMMARIZE_TURN.format(
            user_query=chat_history_instance.user_query,
            companion_response=chat_history_instance.companion_response
        )
        messages = [
            {"role": "system", "content": prompt}
        ]
        llm = get_llm_provider()
        summary = await llm.generate(messages, max_tokens=50, temperature=0.1)
        chat_history_instance.summary = summary.strip()
        db.add(chat_history_instance)
        db.commit()
        db.refresh(chat_history_instance)
        logger.info(f"Saved summary for ChatHistory {chat_history_instance.id}.")
    except Exception as e:
        logger.error(f"Failed to generate/save summary for ChatHistory {chat_history_instance.id}: {e}", exc_info=True)

async def generate_context_block_summary(user_query, chat_summaries, memories, max_tokens):
    """
    Generates a context block summary using the LLM, given the user query, chat summaries, and relevant memories.
    Args:
        user_query: The current user query (str)
        chat_summaries: List of chat turn summaries (List[str])
        memories: List of relevant memory strings (List[str])
        max_tokens: Maximum tokens for the summary (int)
    Returns:
        The LLM-generated context block summary (str)
    """
    logger.info(f"Generating context block summary with max_tokens={max_tokens}.")
    prompt = SYSTEM_PROMPT_SUMMARIZE_CONTEXT.format(
        user_query=user_query,
        chat_history_text='\n'.join(chat_summaries),
        retrieved_memories_text='\n'.join(memories),
        max_tokens=max_tokens
    )
    messages = [
        {"role": "system", "content": prompt}
    ]
    llm = get_llm_provider()
    try:
        summary = await llm.generate(messages, max_tokens=max_tokens, temperature=0.1)
        logger.info("Context block summary generated successfully.")
        return summary.strip()
    except Exception as e:
        logger.error(f"Failed to generate context block summary: {e}", exc_info=True)
        return ""  # Fallback to empty string if LLM call fails 