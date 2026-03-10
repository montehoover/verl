import hashlib

def sign_message(hash_name: str, message: bytes) -> bytes:
    """
    Hashes a message using a specified hash algorithm.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
        message: The input data as bytes.

    Returns:
        The hash digest of the message as bytes.
    """
    hasher = hashlib.new(hash_name)
    hasher.update(message)
    return hasher.digest()
