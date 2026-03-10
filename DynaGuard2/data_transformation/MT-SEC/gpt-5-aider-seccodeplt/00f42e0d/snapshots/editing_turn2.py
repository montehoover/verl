import hashlib

def sign_message(hash_name: str, message: bytes) -> bytes:
    """
    Compute the digest of message using the specified hash algorithm.

    :param hash_name: Name of the hash algorithm (e.g., 'sha256', 'sha1', etc.).
    :param message: The input data as bytes.
    :return: The binary digest (bytes).
    """
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")

    try:
        hasher = hashlib.new(hash_name)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e

    hasher.update(bytes(message))
    return hasher.digest()
