import hashlib
from typing import Union


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using the specified algorithm.

    Args:
        algorithm_name: The name of the cryptographic algorithm (case-insensitive).
        user_password: The password to be hashed as bytes.

    Returns:
        The computed hash value as bytes.

    Raises:
        ValueError: If the given hash algorithm is not supported.
    """
    if not isinstance(user_password, (bytes, bytearray, memoryview)):
        raise TypeError("user_password must be a bytes-like object")

    algo_normalized = algorithm_name.lower()
    supported_algorithms = {name.lower() for name in hashlib.algorithms_available}

    if algo_normalized not in supported_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")

    # Create the hasher and compute digest.
    # For SHAKE algorithms (XOF), a digest length is required.
    if algo_normalized in ("shake_128", "shake_256"):
        # Choose sensible default output sizes:
        # - SHAKE128: 32 bytes
        # - SHAKE256: 64 bytes
        digest_len = 32 if algo_normalized == "shake_128" else 64
        hasher = hashlib.new(algo_normalized)
        hasher.update(user_password)
        return hasher.digest(digest_len)

    hasher = hashlib.new(algo_normalized)
    hasher.update(user_password)
    return hasher.digest()
