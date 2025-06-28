"""
Encryption utilities for secure content storage.
Uses Fernet (symmetric encryption) from the cryptography library.
"""
from cryptography.fernet import Fernet
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Initialize Fernet with the key from settings
fernet = Fernet(settings.ENCRYPTION_KEY.encode())

def encrypt_content(content: str) -> bytes:
    """
    Encrypt content using Fernet symmetric encryption.
    """
    try:
        return fernet.encrypt(content.encode())
    except Exception as e:
        logger.error(f"Error encrypting content: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to encrypt content: {str(e)}")

def decrypt_content(encrypted_content: bytes) -> str:
    """
    Decrypt content using Fernet symmetric encryption.
    """
    try:
        return fernet.decrypt(encrypted_content).decode()
    except Exception as e:
        logger.error(f"Error decrypting content: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to decrypt content: {str(e)}") 