"""
Hash utilities.

This module exposes a single function, compute_digest, that computes a
cryptographic digest of the provided content using a requested algorithm.
To promote security, it rejects weak algorithms (e.g., MD5, SHA-1) and
algorithms that require an explicit digest length (SHAKE variants), which
are not supported by this interface.
"""

import hashlib

# Algorithms that are considered weak or unsafe for general cryptographic use.
# These are explicitly rejected even if present in the local OpenSSL build.
UNSAFE_ALGORITHMS = {"md5", "sha1", "md5-sha1"}


def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Compute the digest of the given content using the specified hash algorithm.

    Arguments:
        algorithm_name: The name of the hash algorithm to use. The value is
            case-insensitive (e.g., "SHA256", "sha256").
        content: The input data to hash as a bytes-like object.

    Returns:
        The computed hash as raw bytes (not a hexadecimal string).

    Raises:
        ValueError: If the algorithm name is not a string, the algorithm is
            unavailable on this system, or the algorithm is unsafe or
            unsupported by this interface.

    Notes:
        - Weak algorithms (MD5, SHA-1, and MD5-SHA1) are explicitly rejected.
        - SHAKE algorithms (XOFs) require a digest length and are therefore
          unsupported by this fixed-length digest interface.
    """
    # Validate the algorithm name type early for clearer error messages.
    if not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a string.")

    # Normalize the algorithm name to a common case for comparison.
    alg = algorithm_name.lower()

    # Reject weak/unsafe algorithms even if available in hashlib.
    if alg in UNSAFE_ALGORITHMS:
        raise ValueError(f"Unsafe hash algorithm: {algorithm_name}")

    # Reject algorithms that require an explicit digest length (XOFs).
    if alg in {"shake_128", "shake_256"}:
        raise ValueError(
            "Algorithm requires a digest length and is unsupported: "
            f"{algorithm_name}"
        )

    # Ensure the requested algorithm is actually available in this runtime.
    available = {name.lower() for name in hashlib.algorithms_available}
    if alg not in available:
        raise ValueError(f"Unavailable hash algorithm: {algorithm_name}")

    # Construct the hash object and compute the digest.
    try:
        hasher = hashlib.new(alg)
    except Exception as exc:
        # Normalize any underlying errors as ValueError per the contract.
        raise ValueError(
            "Unavailable or unsupported hash algorithm: "
            f"{algorithm_name}"
        ) from exc

    # Feed the content into the hasher and return the raw digest bytes.
    hasher.update(content)
    return hasher.digest()
