import hashlib
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the cryptographic hash of a byte sequence using the specified algorithm.
    
    This function generates a hash digest for the given content using the specified
    hashing algorithm. It explicitly prevents the use of weak cryptographic algorithms
    like MD5 and SHA-1 for security reasons.
    
    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        content: The byte-like object to be hashed.
    
    Returns:
        The computed hash digest as bytes.
    
    Raises:
        ValueError: If the algorithm is either unsafe (MD5, SHA-1) or not available
                    in the hashlib module.
    
    Example:
        >>> digest = compute_digest('sha256', b'Hello, World!')
        >>> len(digest)
        32
    """
    # Log the function call with algorithm name and content size
    logger.debug(f"compute_digest called with algorithm='{algorithm_name}', content_size={len(content)} bytes")
    
    # Define set of cryptographically weak algorithms that should not be used
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    # Validate that the requested algorithm is not in the unsafe list
    if algorithm_name.lower() in unsafe_algorithms:
        logger.warning(f"Attempted to use unsafe algorithm: '{algorithm_name}'")
        error_msg = f"Algorithm '{algorithm_name}' is unsafe and should not be used"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Attempt to create a hash object with the specified algorithm
    try:
        hash_obj = hashlib.new(algorithm_name, content)
        digest = hash_obj.digest()
        logger.info(f"Successfully computed {algorithm_name} digest, size={len(digest)} bytes")
        return digest
    except ValueError as e:
        # Re-raise with a more descriptive error message
        error_msg = f"Algorithm '{algorithm_name}' is not available"
        logger.error(f"{error_msg}. Original error: {str(e)}")
        raise ValueError(error_msg)
