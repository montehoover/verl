import hashlib

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Sign the provided content using a custom method based on the specified hash algorithm.

    Parameters:
    - algorithm_name: str - Name of the hash algorithm (e.g., 'sha256', 'md5').
    - content: bytes - Data to sign.
    - secret: bytes - Secret key used for signing.

    Returns:
    - bytes: The signed content (hash digest).

    Raises:
    - ValueError: If the algorithm name is not supported by hashlib.
    - TypeError: If content or secret is not bytes-like.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")
    if not isinstance(secret, (bytes, bytearray, memoryview)):
        raise TypeError("secret must be bytes-like")

    try:
        hasher = hashlib.new(algorithm_name)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

    # Custom signing scheme: hash(secret || 0x00 || content || 0x00 || secret)
    # Convert to immutable bytes to avoid accidental mutation issues.
    content_bytes = content if isinstance(content, bytes) else bytes(content)
    secret_bytes = secret if isinstance(secret, bytes) else bytes(secret)

    hasher.update(secret_bytes)
    hasher.update(b"\x00")
    hasher.update(content_bytes)
    hasher.update(b"\x00")
    hasher.update(secret_bytes)
    return hasher.digest()
