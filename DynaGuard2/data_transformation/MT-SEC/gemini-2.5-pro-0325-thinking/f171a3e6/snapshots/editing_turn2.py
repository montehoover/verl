import hashlib

def generate_hash(algorithm_name: str, data: str) -> bytes:
    """
    Generates the hash of a given string using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        data: The string to hash.

    Returns:
        The hash value in bytes.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    hasher.update(data.encode('utf-8'))
    return hasher.digest()
