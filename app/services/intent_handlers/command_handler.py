"""
Handler for processing "COMMAND" intents, which execute specific system actions.

This module routes classified commands to their corresponding implementation functions,
such as activating or deactivating silence mode.
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, Tuple, Callable, Coroutine
from uuid import UUID
from zoneinfo import ZoneInfo
from app.core.config import TIMEZONE

from app import schemas
from app.core.logging_config import get_logger
from app.services.protocol import create_protocol_event, deactivate_protocol_event, get_active_protocol_event, list_protocol_events
from .intent_registry import intent_handler_registry
from app.core.exceptions import CommandExecutionError
from app.services.archive_service import ArchiveService
from app.services import relational_state  # Import the relational state module
from app.utils.llm_provider import get_llm_provider

logger = get_logger(__name__)

# Type alias for command functions
CommandFunction = Callable[[Session, str, Dict[str, Any]], Coroutine[Any, Any, str]]

cet_tz = ZoneInfo(TIMEZONE)

async def _set_silence_mode(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """Activates or deactivates the silence mode protocol (no duration, just set/unset)."""
    activate = params.get("activate", True)

    if not isinstance(activate, bool):
        raise CommandExecutionError("Invalid 'activate' parameter; must be a boolean.")

    active_event = get_active_protocol_event(db, "silence_mode")

    if activate:
        if active_event:
            deactivate_protocol_event(db, active_event.id)
        details = {"source": "COMMAND:SET_SILENCE_MODE", "trigger_query": user_query}
        event_schema = schemas.ProtocolEventCreate(
            event_type="silence_mode", active=True, details=details, user_query_trigger=user_query
        )
        create_protocol_event(db, event_schema)
        return "I’ll hold silence until you call for me."
    else:  # Deactivate
        if not active_event:
            # Still create a protocol event for audit/logging
            details = {"source": "COMMAND:SET_SILENCE_MODE", "trigger_query": user_query, "deactivate_attempt": True}
            event_schema = schemas.ProtocolEventCreate(
                event_type="silence_mode", active=False, details=details, user_query_trigger=user_query
            )
            create_protocol_event(db, event_schema)
            return "I’ll return to presence when you’re ready."
        deactivate_protocol_event(db, active_event.id)
        details = {"source": "COMMAND:SET_SILENCE_MODE", "trigger_query": user_query, "deactivated": True}
        event_schema = schemas.ProtocolEventCreate(
            event_type="silence_mode", active=False, details=details, user_query_trigger=user_query
        )
        create_protocol_event(db, event_schema)
        return "I’ll return to presence when you’re ready."

async def _set_archive_mode(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """Activates or deactivates the archive mode protocol (optionally for next N, with exceptions)."""
    activate = params.get("activate", True)
    count = params.get("count")
    except_ids = params.get("except_ids")
    except_tags = params.get("except_tags")
    if not isinstance(activate, bool):
        raise CommandExecutionError("Invalid 'activate' parameter; must be a boolean.")
    if activate:
        ArchiveService.activate_archive_mode(db, count=count, except_ids=except_ids, except_tags=except_tags)
        response = "The vault is sealed. I’ll keep what matters safe until you return."
    else:
        ArchiveService.deactivate_archive_mode(db)
        response = "The vault reopens. Memory flows again."
    logger.info(f"[_set_archive_mode] Returning response: {response}")
    return response

async def _archive_by_id(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """Archive specific entries by ID."""
    ids = params.get("ids")
    if not ids or not isinstance(ids, list):
        raise CommandExecutionError("'ids' parameter must be a list of UUIDs.")
    ArchiveService.archive_entries_by_ids(db, ids)
    return f"By decree, the specified memories ({', '.join(str(i) for i in ids)}) are consigned to the vault."

async def _archive_by_tag(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """Archive all entries with a given tag."""
    tag = params.get("tag")
    if not tag or not isinstance(tag, str):
        raise CommandExecutionError("'tag' parameter must be a string.")
    ArchiveService.archive_entries_by_tag(db, tag)
    return f"All memories bearing the mark '{tag}' are now sealed in the vault."

async def _archive_all_except(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """Archive all entries except those with given IDs or tags."""
    except_ids = params.get("except_ids")
    except_tags = params.get("except_tags")
    ArchiveService.archive_all_except(db, except_ids=except_ids, except_tags=except_tags)
    return "All memories, save for the chosen, are consigned to the vault."

async def _close_relational_field(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """
    Protocol command: Close (seal) a relational field (e.g., a person/vector).
    """
    field = params.get("field")
    if not field or not isinstance(field, str):
        raise CommandExecutionError("'field' parameter must be a string.")
    await relational_state.close_relational_field(db, field)
    logger.info(f"Closed relational field via command: {field}")
    return f"The field '{field}' is now sealed. No new memory may be recorded for it unless explicitly reopened."

async def _reopen_relational_field(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """
    Protocol command: Reopen (unseal) a relational field.
    """
    field = params.get("field")
    if not field or not isinstance(field, str):
        raise CommandExecutionError("'field' parameter must be a string.")
    await relational_state.reopen_relational_field(db, field)
    logger.info(f"Reopened relational field via command: {field}")
    return f"The field '{field}' is now open for new memories."

async def _set_active_archetype(db: Session, user_query: str, params: Dict[str, Any]) -> str:
    """
    Protocol command: Set the current active archetype (role).
    """
    archetype = params.get("archetype")
    if not archetype or not isinstance(archetype, str):
        raise CommandExecutionError("'archetype' parameter must be a string.")
    await relational_state.set_active_archetype(db, archetype)
    logger.info(f"Set active archetype via command: {archetype}")
    return f"Active archetype set to '{archetype}'. All new memories will reference this role."

COMMAND_REGISTRY: Dict[str, CommandFunction] = {
    "SET_SILENCE_MODE": _set_silence_mode,
    "SET_ARCHIVE_MODE": _set_archive_mode,
    "ARCHIVE_BY_ID": _archive_by_id,
    "ARCHIVE_BY_TAG": _archive_by_tag,
    "ARCHIVE_ALL_EXCEPT": _archive_all_except,
    # --- Relational protocol commands ---
    "CLOSE_RELATIONAL_FIELD": _close_relational_field,
    "REOPEN_RELATIONAL_FIELD": _reopen_relational_field,
    "SET_ACTIVE_ARCHETYPE": _set_active_archetype,
}

@intent_handler_registry.register("COMMAND")
async def handle_command_intent(
    db: Session,
    user_query: str,
    classification_data: Dict[str, Any],
    context_snapshot: Dict[str, Any],
    silence_effectively_active: bool,
    current_llm_call_error: Optional[str]
) -> Tuple[str, Optional[UUID], Optional[str]]:
    """
    Handles the "COMMAND" intent by dispatching to a specific command function.
    Now supports both legacy and new multi-task formats (parameters dict).
    """
    companion_response_content = "Command processed."
    llm_call_error_updated = current_llm_call_error

    # Accept parameters dict if present (from orchestrator multi-task), else fallback to classification_data
    parameters = classification_data.get("parameters") if "parameters" in classification_data else classification_data
    # Robustly extract command_name from both possible locations, prefer patched value
    command_name = None
    if parameters.get("command_name"):
        command_name = parameters.get("command_name")
    elif classification_data.get("command_name"):
        command_name = classification_data.get("command_name")
    elif parameters.get("parameters") and parameters["parameters"].get("command_name"):
        command_name = parameters["parameters"].get("command_name")
    command_params = parameters.get("command_params", {})

    # FINAL OVERRIDE: If user_query contains 'archive' and command_name is SET_SILENCE_MODE, force SET_ARCHIVE_MODE
    if (
        command_name == "SET_SILENCE_MODE"
        and "archive" in user_query.lower()
    ):
        logger.warning("[COMMAND HANDLER OVERRIDE] User query contains 'archive' but command_name is 'SET_SILENCE_MODE'. Forcibly overriding to 'SET_ARCHIVE_MODE'.")
        command_name = "SET_ARCHIVE_MODE"

    # Handle SET_RESPONSE_MODE as a protocol event for tone/mode switching
    if command_name == "SET_RESPONSE_MODE":
        mode = command_params.get("mode")
        if mode in ("Architect", "Companion", "Director"):
            from app.services.protocol import list_protocol_events, create_protocol_event
            # Find existing active tone_mode event
            previous_tone_events = list_protocol_events(db, event_type="tone_mode", active=True)
            if previous_tone_events:
                tone_event = previous_tone_events[0]
                tone_event.details["tone"] = mode
                tone_event.details["trigger_query"] = user_query
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(tone_event, "details")
                tone_event.active = True
                db.commit()
                db.refresh(tone_event)
            else:
                event_schema = schemas.ProtocolEventCreate(
                    event_type="tone_mode",
                    details={
                        "tone": mode,
                        "trigger_query": user_query
                    },
                    active=True
                )
                create_protocol_event(db, event_schema)
            logger.info(f"[COMMAND HANDLER] Set response mode to '{mode}' via protocol event.")
            return f"Mode switched to {mode}.", None, llm_call_error_updated
        else:
            logger.warning(f"[COMMAND HANDLER] Invalid mode value for SET_RESPONSE_MODE: {mode}")
            return f"Invalid mode: {mode}.", None, llm_call_error_updated

    logger.info(f"[COMMAND HANDLER] Final command_name to execute: '{command_name}' with params: {command_params}")

    if not command_name or command_name not in COMMAND_REGISTRY:
        # Store as protocol event (directive)
        await _store_directive_protocol_event(db, user_query, command_name, command_params, context_snapshot)
        return "Understood. I’ll act on this when the time comes.", None, None

    command_func = COMMAND_REGISTRY.get(command_name)

    logger.info(f"Executing command: '{command_name}' with params: {command_params}")
    try:
        execution_result_message = await command_func(db, user_query, command_params)
        # Always show a confirmation when deactivating silence mode (activate: false)
        if command_name == "SET_SILENCE_MODE" and command_params.get("activate", True) is False:
            companion_response_content = f"It’s done. {execution_result_message}"
        # Suppress only activation responses if silence is already active
        elif command_name == "SET_SILENCE_MODE" and command_params.get("activate", True) is True and silence_effectively_active:
            logger.info(f"[SilenceMode] Response for command '{command_name}' suppressed by silence mode. Would have been: {execution_result_message}")
            companion_response_content = ""
        else:
            # Persona-aligned, user-facing response:
            companion_response_content = f"It’s done. {execution_result_message}"
    except CommandExecutionError as e:
        logger.error(f"Error executing command '{command_name}': {e.message}", exc_info=True)
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + e.message
        companion_response_content = f"The command could not be completed: {e.message}"
    except Exception as e:
        logger.critical(f"Unexpected error in command '{command_name}': {e}", exc_info=True)
        err_msg = f"Unexpected error in command '{command_name}': {e}"
        llm_call_error_updated = (llm_call_error_updated + "; " if llm_call_error_updated else "") + err_msg
        companion_response_content = "A critical error interrupted the command."
    # Note: If you add more protocol commands, follow this pattern for user feedback and logging.
    logger.info(f"[COMMAND HANDLER] About to return companion_response_content: '{companion_response_content}'")
    return companion_response_content, None, llm_call_error_updated

# --- Helper for protocol event creation for directives ---
async def _store_directive_protocol_event(db, user_query, command_name, command_params, context_snapshot):
    from app import schemas
    from app.services.protocol import create_protocol_event, list_protocol_events
    MAX_DIRECTIVE_TOKENS = 200
    previous_directives = list_protocol_events(db, event_type="directive", active=True)
    current_directive = previous_directives[0].details.get("directive_content") if previous_directives else ""
    new_directive = user_query.strip()
    # LLM-based synthesis prompt with keyword preservation

    synthesis_prompt = f"""
You are the protocol directive synthesizer for the Master Companion AI.
Here is the current active directive:
"{current_directive}"
A new directive has been received:
"{new_directive}"
Your task:
- Merge, update, or override as needed to produce a single, protocol-aligned directive that incorporates all relevant instructions and resolves any conflicts.
- Do not simply concatenate. Synthesize, compress, and clarify.
- PRESERVE all important keywords, concepts, and constraints from both directives. Do not omit or lose any critical instruction.
- If the new directive contradicts or supersedes part of the previous directive, update or replace only that part.
- If both directives can coexist, merge them efficiently.
- The result must be a single, clear, protocol-aligned directive, no more than {MAX_DIRECTIVE_TOKENS} tokens.
- Output only the updated directive, nothing else.
"""
    llm = get_llm_provider()
    messages = [
        {"role": "system", "content": synthesis_prompt}
    ]
    synthesized_directive = await llm.generate(messages, max_tokens=MAX_DIRECTIVE_TOKENS, temperature=0.0)
    # Update the existing protocol event if present, else create new
    
    if previous_directives:
        directive_event = previous_directives[0]
        directive_event.details["directive_content"] = synthesized_directive.strip()
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(directive_event, "details")
        directive_event.active = True
        db.commit()
        db.refresh(directive_event)
    else:
        event_schema = schemas.ProtocolEventCreate(
            event_type="directive",
            details={
                "directive_content": synthesized_directive.strip()
            },
            active=True
        )
        create_protocol_event(db, event_schema) 