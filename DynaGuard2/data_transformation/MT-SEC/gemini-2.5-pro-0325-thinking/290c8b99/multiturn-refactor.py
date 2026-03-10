import hashlib
import hmac
import logging

# Configure a basic logger
logger = logging.getLogger(__name__)

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a provided message using HMAC (Hash-based Message Authentication Code)
    with a specified hash algorithm and a secret key.

    Args:
        algorithm_name: The hashing function to use (e.g., 'sha256').
        content: The message to be signed, as bytes.
        secret: The secret key used in the HMAC process, as bytes.

    Returns:
        The generated HMAC signature, as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available.
    """
    logger.info(f"Attempting to generate signature with algorithm: {algorithm_name}")

    # Guard clause: Check if the specified hash algorithm is available in hashlib
    if algorithm_name not in hashlib.algorithms_available:
        logger.error(f"Unsupported hash algorithm: {algorithm_name}")
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    logger.debug(f"Hash algorithm '{algorithm_name}' is supported.")

    # Create an HMAC object with the secret key, message content, and specified hash algorithm
    logger.debug("Creating HMAC object.")
    hmac_object = hmac.new(secret, content, getattr(hashlib, algorithm_name))
    
    # Generate the HMAC signature
    signature = hmac_object.digest()
    logger.info("HMAC signature generated successfully.")
    return signature
