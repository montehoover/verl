import hashlib

def compute_hash(data: str, algorithm: str) -> str:
    """
    Compute the hash value of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value as a string.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()
