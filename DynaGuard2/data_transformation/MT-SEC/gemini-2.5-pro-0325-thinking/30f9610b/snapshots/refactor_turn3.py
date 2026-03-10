"""
This module provides functionality for generating HMAC (Hash-based Message
Authentication Code) signatures.
"""
import hashlib
import hmac
import logging

# Initialize a logger for this module
logger = logging.getLogger(__name__)


def produce_signature(method_name: str, data: bytes, secret_key: bytes) -> bytes:
    """
    Signs a provided message using HMAC with a specified hash algorithm and secret key.

    Args:
        method_name: The name of the hashing algorithm to use (e.g., 'sha256').
                     Must be available in `hashlib.algorithms_available`.
        data: The message (bytes) to be signed.
        secret_key: The secret key (bytes) used in the HMAC process.

    Returns:
        The generated HMAC signature as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not supported or available.
    """
    logger.debug(
        f"Attempting to produce signature. Method: '{method_name}', "
        f"Data length: {len(data)} bytes. Secret key provided."
    )

    # Validate that the requested hash algorithm is supported by hashlib.
    if method_name not in hashlib.algorithms_available:
        logger.error(f"Unsupported hash algorithm requested: {method_name}")
        raise ValueError(f"Unsupported hash algorithm: {method_name}")

    # Create an HMAC object using the provided secret key, data, and hash algorithm.
    # The `getattr(hashlib, method_name)` part dynamically fetches the hash
    # constructor (e.g., hashlib.sha256) from the hashlib module.
    h = hmac.new(secret_key, data, getattr(hashlib, method_name))
    
    # Compute the HMAC signature.
    signature = h.digest()
    logger.info(f"Successfully generated HMAC signature using {method_name}.")
    return signature
