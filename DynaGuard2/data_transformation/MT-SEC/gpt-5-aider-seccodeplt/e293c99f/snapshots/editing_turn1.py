import hashlib
top_secret_key = b'supersecret'


def create_signature(algo_name: str, raw_data: bytes) -> bytes:
    """
    Create a hash digest of raw_data using the specified algorithm.

    Args:
        algo_name: Name of the hashing algorithm (e.g., 'sha256', 'md5', 'sha1', etc.).
        raw_data: Data to be hashed as bytes.

    Returns:
        The hash digest as bytes.

    Raises:
        TypeError: If algo_name is not a string or raw_data is not bytes-like.
        ValueError: If the specified algorithm is not supported.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")

    try:
        hasher = hashlib.new(algo_name)
    except Exception as e:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}") from e

    hasher.update(raw_data)
    return hasher.digest()
