class CoreApplicationException(Exception):
    """Base class for the application's custom exceptions."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

# --- Service-Related Exceptions ---
class ServiceError(CoreApplicationException):
    """Base class for exceptions related to external services."""
    def __init__(self, service_name: str, message: str, details: dict = None):
        self.service_name = service_name
        super().__init__(f"Error with service '{service_name}': {message}", details=details)

class LLMProviderError(ServiceError):
    """Raised when an LLM provider call fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(service_name="LLMProvider", message=message, details=details)

# --- Data-Related Exceptions ---
class DataError(CoreApplicationException):
    """Base class for exceptions related to data processing, validation, or access."""
    pass

class DatabaseOperationError(DataError):
    """Raised when a database operation fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Database operation failed: {message}", details=details)

class DocumentProcessingError(ServiceError):
    """Exception for errors during document processing (e.g., parsing, splitting)."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(service_name="DocumentProcessor", message=message, details=details)

# --- Configuration and Execution Exceptions ---
class ConfigurationError(CoreApplicationException):
    """Raised for configuration-related problems."""
    pass

class TaskOrchestrationError(CoreApplicationException):
    """Raised for errors during the task orchestration process (e.g., planning failure)."""
    pass

class SubTaskExecutionError(TaskOrchestrationError):
    """Raised when a specific sub-task fails during execution by its handler."""
    def __init__(self, sub_task_id: str, intent: str, message: str, details: dict = None):
        self.sub_task_id = sub_task_id
        self.intent = intent
        super().__init__(f"Sub-task '{sub_task_id}' (Intent: {intent}) failed: {message}", details=details)

class CommandExecutionError(ServiceError):
    """Exception for errors during command execution in the command_handler."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(service_name="CommandHandler", message=message, details=details)

# --- Input/Validation Exceptions ---
class InputTooLongError(CoreApplicationException):
    """
    Raised when the user query and context exceed the LLM's maximum input token limit.
    Should be caught and handled to provide a protocol-aligned, user-friendly error message.
    """
    def __init__(self, message: str = None, details: dict = None):
        msg = message or "Your query and context exceed the maximum allowed input size. Please shorten your request."
        super().__init__(msg, details=details) 