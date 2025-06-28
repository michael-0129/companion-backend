"""
Intent Handlers Package.

This package uses a decorator-based system to register intent handlers.
Importing the handler modules here ensures that their decorators are processed
at application startup, populating the central registry.
"""

from .intent_registry import intent_handler_registry, IntentHandler

# Import handler modules to ensure their decorators are registered.
from . import query_handler
from . import memory_handler
from . import command_handler

# Re-export the registry object for use in other parts of the application.
__all__ = [
    "intent_handler_registry",
    "IntentHandler",
] 