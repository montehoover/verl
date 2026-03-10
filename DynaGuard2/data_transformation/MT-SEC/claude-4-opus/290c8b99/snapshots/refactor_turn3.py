import hashlib
import hmac
import logging


# Set up logger for this module
logger = logging.getLogger(__name__)


def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.
    
    Args:
        algorithm_name: The hashing function to use
        content: The message to be signed
        secret: The secret key used in the HMAC process
        
    Returns:
        The generated HMAC signature as bytes
        
    Raises:
        ValueError: If the specified hash algorithm is not supported or available
    """
    # Log function entry with algorithm name
    logger.debug(f"Generating signature using algorithm: {algorithm_name}")
    
    # Guard clause: Check if algorithm exists in hashlib
    if not hasattr(hashlib, algorithm_name):
        error_message = f"Unsupported hash algorithm: {algorithm_name}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Retrieve the hash function from hashlib module
    hash_function = getattr(hashlib, algorithm_name)
    logger.debug(f"Successfully retrieved hash function for: {algorithm_name}")
    
    # Create HMAC object with the secret key and specified hash algorithm
    hmac_object = hmac.new(secret, content, hash_function)
    logger.debug("HMAC object created successfully")
    
    # Generate and return the HMAC signature as bytes
    signature_bytes = hmac_object.digest()
    logger.debug(f"Generated signature of {len(signature_bytes)} bytes")
    
    return signature_bytes
