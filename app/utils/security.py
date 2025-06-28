import os
import base64
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

from app.core.logging_config import get_logger
from app.core.exceptions import ConfigurationError

logger = get_logger(__name__)
load_dotenv()

def _initialize_fernet() -> Fernet:
    """
    Initializes the Fernet instance for encryption/decryption.

    Retrieves the encryption key from environment variables, validates it,
    and returns a Fernet instance.

    Returns:
        A configured Fernet instance.

    Raises:
        ConfigurationError: If the ENCRYPTION_KEY is missing, invalid,
                            or does not decode to the required 32 bytes.
    """
    key_str = os.getenv("ENCRYPTION_KEY")
    if not key_str:
        logger.critical("ENCRYPTION_KEY is not set in the environment. Application cannot start.")
        raise ConfigurationError("ENCRYPTION_KEY is missing. Please set it in your .env file.")

    try:
        # Fernet keys are URL-safe base64-encoded 32-byte keys.
        key_bytes = base64.urlsafe_b64decode(key_str)
    except (base64.binascii.Error, ValueError) as e:
        logger.critical(f"ENCRYPTION_KEY is not valid base64: {e}")
        raise ConfigurationError(f"ENCRYPTION_KEY is not a valid base64 string: {e}") from e

    if len(key_bytes) != 32:
        logger.critical(f"ENCRYPTION_KEY must be 32 bytes long after decoding, but was {len(key_bytes)} bytes.")
        raise ConfigurationError("Decoded ENCRYPTION_KEY must be exactly 32 bytes long.")

    logger.info("✅ Encryption key loaded and validated successfully.")
    return Fernet(key_str.encode())

# Global Fernet instance, initialized on module load.
# The application will fail to start if the key is invalid.
fernet = _initialize_fernet()

def encrypt_content(content: str) -> bytes:
    """Encrypts a string using the global Fernet instance."""
    return fernet.encrypt(content.encode('utf-8'))

def decrypt_content(token: bytes) -> str:
    """
    Decrypts a token using the global Fernet instance.
    
    Raises:
        ValueError: If the token is invalid or tampered with.
    """
    try:
        return fernet.decrypt(token).decode('utf-8')
    except InvalidToken as e:
        logger.error("Decryption failed: Invalid token.", exc_info=True)
        raise ValueError("Invalid token - may be corrupted or tampered with") from e
