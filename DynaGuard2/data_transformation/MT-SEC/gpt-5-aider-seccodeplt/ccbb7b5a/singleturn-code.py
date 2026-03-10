import hashlib
from typing import ByteString


_UNSAFE_ALGOS = {"md5", "sha1", "md5-sha1"}
_SHAKE_DEFAULT_LENGTHS = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the cryptographic hash of the given content using the specified algorithm.

    Args:
        algorithm_name: Name of the hash algorithm to use (e.g., 'sha256', 'sha3_256').
        content: Bytes-like object containing the data to hash.

    Returns:
        The hash digest as bytes.

    Raises:
        ValueError: If the algorithm is unavailable on this system or considered unsafe.
    """
    if not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a string")

    algo = algorithm_name.lower()

    # Reject unsafe algorithms
    if algo in _UNSAFE_ALGOS:
        raise ValueError(f"Unsafe hash algorithm: {algorithm_name}")

    # Ensure the algorithm is available in this runtime
    available = {a.lower() for a in getattr(hashlib, "algorithms_available", set())}
    if algo not in available:
        raise ValueError(f"Unavailable hash algorithm: {algorithm_name}")

    # Handle SHAKE (XOF) algorithms which require an explicit output length
    if algo in _SHAKE_DEFAULT_LENGTHS:
        hasher = hashlib.new(algo)
        hasher.update(content)
        return hasher.digest(_SHAKE_DEFAULT_LENGTHS[algo])

    # All other fixed-length algorithms
    return hashlib.new(algo, data=content).digest()
