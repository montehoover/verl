import hashlib

def generate_signature(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the hash digest of the provided content using the specified algorithm.

    Parameters:
    - algorithm_name: str - Name of the hash algorithm (e.g., 'sha256', 'md5').
    - content: bytes - Data to hash.

    Returns:
    - bytes: The hash digest of the content.

    Raises:
    - ValueError: If the algorithm name is not supported by hashlib.
    - TypeError: If content is not bytes-like.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes")

    hasher = hashlib.new(algorithm_name)
    hasher.update(content)
    return hasher.digest()
