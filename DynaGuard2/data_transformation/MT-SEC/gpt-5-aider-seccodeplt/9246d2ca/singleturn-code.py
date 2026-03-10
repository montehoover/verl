import hashlib
from typing import Union


def encrypt_password(algo_name: str, pass_key: bytes) -> bytes:
    """
    Compute a cryptographic hash for the given password using the specified algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm to use (e.g., 'sha256').
        pass_key: The password to be hashed, as bytes.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the specified algorithm is not supported.
        TypeError: If pass_key is not bytes-like.
    """
    if not isinstance(pass_key, (bytes, bytearray, memoryview)):
        raise TypeError("pass_key must be bytes-like")

    # Normalize algorithm name for comparison (hashlib is generally case-insensitive)
    algo_normalized = algo_name.lower()
    available = {name.lower() for name in hashlib.algorithms_available}

    if algo_normalized not in available:
        raise ValueError(f"Unsupported hash algorithm: {algo_name}")

    # Handle SHAKE (XOF) algorithms which require an explicit digest length
    if algo_normalized in ("shake_128", "shake_256"):
        # Sensible defaults: 32 bytes for shake_128, 64 bytes for shake_256
        default_lengths = {"shake_128": 32, "shake_256": 64}
        h = hashlib.new(algo_normalized)
        h.update(bytes(pass_key))
        return h.digest(default_lengths[algo_normalized])

    # For fixed-length algorithms, return the full digest
    return hashlib.new(algo_normalized, bytes(pass_key)).digest()
