import hashlib
import hmac
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

def validate_hash_algorithm(hash_name: str) -> None:
    """
    Validates that the hash algorithm is available in hashlib.
    
    Args:
        hash_name: str, the name of the hash algorithm
    
    Raises:
        ValueError: when the hash algorithm is not available
    """
    if not hasattr(hashlib, hash_name):
        raise ValueError(f"Hash algorithm '{hash_name}' is not available")

def sign_message(hash_name: str, message: bytes, key: bytes) -> bytes:
    """
    Signs the given message using the specified hash algorithm and a secret key.
    
    Args:
        hash_name: str, the name of the hash algorithm
        message: bytes, the message to be signed
        key: bytes, the secret key used for signing
    
    Returns:
        bytes, the signed message
    
    Raises:
        ValueError: when the hash algorithm is not available
    """
    logger.info(f"Starting message signing with hash algorithm: {hash_name}")
    
    try:
        # Guard clause - validate hash algorithm early
        validate_hash_algorithm(hash_name)
        logger.debug(f"Hash algorithm '{hash_name}' validated successfully")
        
        # Create HMAC object with the validated hash algorithm
        h = hmac.new(key, message, getattr(hashlib, hash_name))
        signature = h.digest()
        
        logger.info(f"Message signed successfully with {hash_name}, signature length: {len(signature)} bytes")
        return signature
        
    except ValueError as e:
        logger.error(f"Failed to sign message: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during message signing: {str(e)}")
        raise
