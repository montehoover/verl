import hashlib
from typing import Union


def hash_data(algo_name: str, content_data: Union[bytes, bytearray, memoryview]) -> bytes:
    """
    Generate a cryptographic hash of the given content_data using the specified algorithm.

    Args:
        algo_name: Name of the hash algorithm to use (e.g., 'sha256', 'sha3_256', 'blake2b').
        content_data: Input data to be hashed as bytes (bytes-like is accepted and converted).

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or disallowed (e.g., md5, sha1).
        TypeError: If content_data is not bytes-like.
    """
    if not isinstance(content_data, (bytes, bytearray, memoryview)):
        raise TypeError("content_data must be bytes or bytes-like (bytearray, memoryview)")

    algo = algo_name.lower().strip()
    available = {a.lower() for a in hashlib.algorithms_available}
    disallowed = {"md5", "sha1", "md5-sha1"}

    if algo in disallowed:
        raise ValueError(f"Disallowed hash algorithm: {algo}")

    if algo not in available:
        raise ValueError(f"Unavailable hash algorithm: {algo}")

    data_bytes = bytes(content_data)

    # Handle variable-length XOF algorithms
    if algo in ("shake_128", "shake_256"):
        # Choose sensible default digest lengths:
        # - shake_128 -> 32 bytes
        # - shake_256 -> 64 bytes
        digest_len = 32 if algo == "shake_128" else 64
        return hashlib.new(algo, data_bytes).digest(digest_len)

    # All fixed-length algorithms
    return hashlib.new(algo, data_bytes).digest()
