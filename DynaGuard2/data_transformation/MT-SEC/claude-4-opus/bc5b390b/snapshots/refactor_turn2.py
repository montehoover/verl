import hashlib


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
    if _is_algorithm_unsafe(algorithm_name):
        raise ValueError(f"Unsafe hash algorithm '{algorithm_name}' is not allowed")
    
    if not _is_algorithm_available(algorithm_name):
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not available")
    
    return _compute_hash(algorithm_name, content)
