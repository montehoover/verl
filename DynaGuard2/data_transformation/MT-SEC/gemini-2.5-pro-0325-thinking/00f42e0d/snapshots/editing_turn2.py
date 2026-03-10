import hashlib

def sign_message(hash_name: str, message: bytes) -> bytes:
    """
    Hashes a message using a specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        message: The input data as bytes.

    Returns:
        The hash digest of the message as bytes.
    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    try:
        hasher = hashlib.new(hash_name)
    except ValueError:
        raise ValueError(f"Hash algorithm '{hash_name}' is not available.")
    hasher.update(message)
    return hasher.digest()
