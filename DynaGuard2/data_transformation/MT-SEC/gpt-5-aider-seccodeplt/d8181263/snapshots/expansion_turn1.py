import hashlib

def hash_data(data: str, algorithm: str) -> str:
    """
    Compute the hash of the input string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The hash algorithm name (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal digest string.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()
