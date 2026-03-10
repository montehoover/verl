import hashlib

def compute_message_hash(message: str, algorithm_name: str) -> bytes:
    """
    Computes the hash of a message string using the specified algorithm.

    Args:
        message: The string to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        The computed hash as bytes.

    Raises:
        ValueError: If the specified algorithm_name is not supported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
    
    hasher.update(message.encode('utf-8'))
    return hasher.digest()
