import hashlib
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def _is_algorithm_unsafe(algorithm_name: str) -> bool:
    """Check if the given algorithm is considered unsafe."""
    unsafe_algorithms = {'md5', 'sha1', 'md5-sha1'}
    return algorithm_name.lower() in unsafe_algorithms


def _is_algorithm_available(algorithm_name: str) -> bool:
    """Check if the given algorithm is available in hashlib."""
    return algorithm_name in hashlib.algorithms_available


def _compute_hash(algorithm_name: str, content: bytes) -> bytes:
    """Compute the hash of the content using the specified algorithm."""
    hash_obj = hashlib.new(algorithm_name, content)
    return hash_obj.digest()


def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """Generate hash of input data using the specified hash algorithm.
    
    Args:
        algorithm_name: The name of the hash algorithm to use.
        content: The input data to be hashed.
        
    Returns:
        The resulting hash of the provided content.
        
    Raises:
        ValueError: If the specified hash algorithm is unavailable or disallowed.
    """
    content_size = len(content)
    logger.info(f"Starting hash generation: algorithm='{algorithm_name}', content_size={content_size} bytes")
    
    if _is_algorithm_unsafe(algorithm_name):
        error_msg = f"Unsafe hash algorithm '{algorithm_name}' is not allowed"
        logger.error(f"Hash generation failed: {error_msg}")
        raise ValueError(error_msg)
    
    if not _is_algorithm_available(algorithm_name):
        error_msg = f"Hash algorithm '{algorithm_name}' is not available"
        logger.error(f"Hash generation failed: {error_msg}")
        raise ValueError(error_msg)
    
    try:
        result = _compute_hash(algorithm_name, content)
        logger.info(f"Hash generation successful: algorithm='{algorithm_name}', content_size={content_size} bytes, hash_size={len(result)} bytes")
        return result
    except Exception as e:
        logger.error(f"Unexpected error during hash computation: algorithm='{algorithm_name}', error='{str(e)}'")
        raise
