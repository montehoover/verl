import hashlib

def generate_signature(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of a given message using a specified algorithm.

    Args:
        algorithm_name: The hash algorithm to utilize (e.g., "sha256", "md5").
        content: The data that needs hashing (bytes).

    Returns:
        The hash digest of the content in bytes.
    """
    hasher = hashlib.new(algorithm_name)
    hasher.update(content)
    return hasher.digest()
