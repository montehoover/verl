import hashlib

def basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hash of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest of the hash as a string.

    Raises:
        ValueError: If the specified hash algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()
