"""
Utilities for hashing data using secure algorithms via Python's hashlib.

This module exposes a single function, `hash_data`, which computes a digest for
the provided bytes using a caller-specified algorithm. Insecure algorithms
(e.g., md5, sha1, md5-sha1) are explicitly disallowed to encourage safe usage.

Notes:
- For extendable-output functions (XOFs) like SHAKE-128 and SHAKE-256, an
  explicit output length is required by hashlib. Reasonable default sizes are
  provided by `_XOF_DEFAULT_SIZES`.
"""

import hashlib

# Algorithms that are considered insecure and must not be used.
# These are disallowed even if present in hashlib.algorithms_available.
_UNSAFE_ALGOS = frozenset({"md5", "sha1", "md5-sha1"})

# Default output sizes (in bytes) for XOF algorithms requiring explicit digest
# length. These defaults aim to provide commonly desired security strengths.
_XOF_DEFAULT_SIZES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    """
    Generate the hash of input data using the specified algorithm.

    Parameters:
        algo_name (str): The name of the hash algorithm to use. This must match
            an algorithm provided by `hashlib.algorithms_available` (case-insensitive),
            and it must not be a disallowed algorithm.
        content_data (bytes): The input data to hash. While the type hint is
            `bytes` for simplicity, bytes-like objects (bytearray, memoryview)
            are accepted at runtime.

    Returns:
        bytes: The hash digest of the provided content.

    Raises:
        ValueError: If the algorithm name is empty, unavailable, or disallowed.
        TypeError: If `content_data` is not a bytes-like object.

    Examples:
        >>> hash_data("sha256", b"hello")
        b'\\x2c\\xf2\\x4d\\xba...'

        >>> hash_data("shake_256", b"data")  # Uses default 64-byte output size
        b'\\x12\\x34...'
    """
    # Validate the algorithm name.
    if not isinstance(algo_name, str) or not algo_name.strip():
        raise ValueError("Algorithm name must be a non-empty string.")

    # Normalize the algorithm name for comparison (hashlib names are case-insensitive).
    algo = algo_name.strip().lower()

    # Reject insecure algorithms outright, even if available.
    if algo in _UNSAFE_ALGOS:
        raise ValueError(
            f"Hash algorithm '{algo_name}' is disallowed due to security concerns."
        )

    # Ensure the requested algorithm is supported in the current environment.
    # Use a case-insensitive comparison by normalizing the available names.
    available_lower = {name.lower() for name in hashlib.algorithms_available}
    if algo not in available_lower:
        raise ValueError(
            f"Hash algorithm '{algo_name}' is not available in this environment."
        )

    # Validate the input data is bytes-like.
    if not isinstance(content_data, (bytes, bytearray, memoryview)):
        raise TypeError("content_data must be a bytes-like object.")

    # Create the hash object and process the data.
    hasher = hashlib.new(algo)
    hasher.update(content_data)

    # Handle extendable-output functions (XOFs) that require an explicit digest length.
    if algo in _XOF_DEFAULT_SIZES:
        return hasher.digest(_XOF_DEFAULT_SIZES[algo])

    # For fixed-length digests, return the standard digest.
    return hasher.digest()
