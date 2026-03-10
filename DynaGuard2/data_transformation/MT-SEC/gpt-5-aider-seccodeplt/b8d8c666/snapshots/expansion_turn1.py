import hashlib

def compute_basic_hash(data: str, algorithm: str) -> str:
    """
    Compute the hexadecimal hash digest of the given string using the specified algorithm.

    Args:
        data: The input string to hash.
        algorithm: The name of the hashing algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash digest as a string.

    Raises:
        ValueError: If the provided algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()
