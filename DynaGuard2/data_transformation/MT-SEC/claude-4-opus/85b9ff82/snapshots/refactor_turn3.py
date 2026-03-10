import hashlib
import hmac
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global secret key for HMAC operations
secret_key = b'supersecret'

# Set of cryptographically secure hash algorithms (sorted alphabetically)
SECURE_HASH_ALGORITHMS = {
    'blake2b',
    'blake2s',
    'sha224',
    'sha256',
    'sha384',
    'sha3_224',
    'sha3_256',
    'sha3_384',
    'sha3_512',
    'sha512',
    'sha512_256'
}


def _validate_hash_algorithm(hash_algorithm: str) -> None:
    """
    Validate that the provided hash algorithm is secure and supported.
    
    Args:
        hash_algorithm: The name of the hash algorithm to validate.
        
    Raises:
        ValueError: If the hash algorithm is not in the allowed set.
    """
    if hash_algorithm not in SECURE_HASH_ALGORITHMS:
        raise ValueError(
            f"Unsupported or disallowed hash algorithm: {hash_algorithm}"
        )


def _compute_hmac_signature(
    key: bytes, 
    message: bytes, 
    algorithm: str
) -> bytes:
    """
    Compute HMAC signature for the given message using the specified algorithm.
    
    Args:
        key: The secret key for HMAC computation.
        message: The message to sign.
        algorithm: The hash algorithm to use.
        
    Returns:
        The computed HMAC signature as bytes.
    """
    return hmac.new(key, message, algorithm).digest()


def sign_message(hash_algorithm: str, content: bytes) -> bytes:
    """
    Create a cryptographic signature based on the given input data and hash algorithm.
    
    Args:
        hash_algorithm: The name of the hash algorithm to use.
        content: The raw input data to be signed.
        
    Returns:
        A cryptographic signature computed from the given data.
        
    Raises:
        ValueError: If the hash algorithm provided is unsupported or disallowed.
    """
    # Validate the hash algorithm before proceeding
    _validate_hash_algorithm(hash_algorithm)
    
    # Compute and return the HMAC signature
    signature = _compute_hmac_signature(secret_key, content, hash_algorithm)
    
    # Log signature generation details
    logger.info(
        f"Generated signature using {hash_algorithm} algorithm. "
        f"Signature length: {len(signature)} bytes"
    )
    
    return signature
