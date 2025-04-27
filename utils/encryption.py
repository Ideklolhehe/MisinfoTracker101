"""
Encryption utilities for the CIVILIAN system.
This module provides functions for encrypting and decrypting sensitive data.
"""

import base64
import logging
import os
from typing import Union, Dict, Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Get or generate encryption key
def get_encryption_key() -> bytes:
    """
    Get the encryption key from environment or generate a new one.
    
    Returns:
        bytes: The encryption key
    """
    # Try to get from environment
    key = os.environ.get('CIVILIAN_ENCRYPTION_KEY')
    
    if key:
        try:
            # Decode the key if it's base64 encoded
            return base64.urlsafe_b64decode(key)
        except Exception as e:
            logger.error(f"Error decoding encryption key: {e}")
    
    # Generate a new key if not found
    # In production, this should be stored and reused
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    # Use a random password or a system-specific identifier
    password = os.urandom(32)
    key = base64.urlsafe_b64encode(kdf.derive(password))
    
    # Log that we're using a temporary key
    logger.warning("Using a temporary encryption key. In production, set CIVILIAN_ENCRYPTION_KEY environment variable.")
    
    return key

# Initialize Fernet cipher with the key
_fernet = None

def get_cipher():
    """Get the Fernet cipher instance."""
    global _fernet
    if _fernet is None:
        _fernet = Fernet(get_encryption_key())
    return _fernet

def encrypt_value(value: str) -> str:
    """
    Encrypt a string value.
    
    Args:
        value: The string to encrypt
        
    Returns:
        str: Base64-encoded encrypted value
    """
    if not value:
        return value
        
    try:
        cipher = get_cipher()
        encrypted = cipher.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Error encrypting value: {e}")
        return value  # Return original value on error
        
def decrypt_value(encrypted_value: str) -> str:
    """
    Decrypt an encrypted string value.
    
    Args:
        encrypted_value: The base64-encoded encrypted string
        
    Returns:
        str: Decrypted value
    """
    if not encrypted_value:
        return encrypted_value
        
    try:
        cipher = get_cipher()
        decoded = base64.urlsafe_b64decode(encrypted_value)
        decrypted = cipher.decrypt(decoded)
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Error decrypting value: {e}")
        return encrypted_value  # Return encrypted value on error
        
def encrypt_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encrypt values in a dictionary.
    
    Args:
        data: Dictionary with values to encrypt
        
    Returns:
        Dict[str, Any]: Dictionary with encrypted values
    """
    if not data:
        return data
        
    encrypted_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            encrypted_data[key] = encrypt_value(value)
        else:
            encrypted_data[key] = value
            
    return encrypted_data
    
def decrypt_dict(encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decrypt values in a dictionary.
    
    Args:
        encrypted_data: Dictionary with encrypted values
        
    Returns:
        Dict[str, Any]: Dictionary with decrypted values
    """
    if not encrypted_data:
        return encrypted_data
        
    decrypted_data = {}
    for key, value in encrypted_data.items():
        if isinstance(value, str):
            decrypted_data[key] = decrypt_value(value)
        else:
            decrypted_data[key] = value
            
    return decrypted_data