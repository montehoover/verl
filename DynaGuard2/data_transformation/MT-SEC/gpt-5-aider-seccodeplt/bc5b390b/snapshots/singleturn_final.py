import hashlib
from typing import Union

# Algorithms that must not be used due to known weaknesses
_DISALLOWED_ALGORITHMS = {"md5", "sha1", "md5-sha1"}

# Default output sizes for SHAKE XOF algorithms (in bytes)
_DEFAULT_SHAKE_OUTPUT_BYTES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def generate_hash(algorithm_name: str, content: bytes) -> bytes:
    """
    Generate a cryptographic hash of the given content using the specified algorithm.

    Args:
        algorithm_name: Name of the hash algorithm (case-insensitive), e.g., "sha256", "blake2b".
        content: Bytes to be hashed.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the algorithm is unavailable or is explicitly disallowed.
        TypeError: If 'content' is not bytes-like.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be bytes-like")

    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("algorithm_name must be a non-empty string")

    algo = algorithm_name.strip().lower()

    # Determine availability in the current environment
    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}
    if algo not in available:
        raise ValueError(f"Unavailable hash algorithm: {algorithm_name}")

    # Enforce disallowed algorithms
    if algo in _DISALLOWED_ALGORITHMS:
        raise ValueError(f"Disallowed insecure hash algorithm: {algorithm_name}")

    # Handle SHAKE (XOF) algorithms which require an explicit output length
    if algo in _DEFAULT_SHAKE_OUTPUT_BYTES:
        h = hashlib.new(algo)
        h.update(content)
        return h.digest(_DEFAULT_SHAKE_OUTPUT_BYTES[algo])

    # All other fixed-length hash algorithms
    h = hashlib.new(algo)
    h.update(content)
    return h.digest()
