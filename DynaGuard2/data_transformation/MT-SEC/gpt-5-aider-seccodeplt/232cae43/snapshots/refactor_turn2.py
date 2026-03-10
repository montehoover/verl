"""
Utilities for computing HMAC signatures.

This module exposes compute_hmac(), a small helper around hmac.new that:
- Validates the requested hash algorithm against hashlib.algorithms_available.
- Normalizes the hash algorithm name in a case-insensitive manner.
"""

import hashlib
import hmac


def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Compute an HMAC signature for the given payload using the provided secret
    and hash algorithm.

    Args:
        hash_algorithm (str):
            Name of the hash function to use (e.g., "sha256"). The check is
            performed case-insensitively against hashlib.algorithms_available.
        payload (bytes):
            The data to sign.
        secret (bytes):
            The secret key used for signing.

    Returns:
        bytes: The HMAC signature as raw bytes.

    Raises:
        ValueError: If the hash function is not supported by hashlib or if
            hash_algorithm is not a string.
    """
    # Validate types of critical parameters early for clearer error messages.
    if not isinstance(hash_algorithm, str):
        raise ValueError("hash_algorithm must be a string")

    # Normalize the requested algorithm to lowercase and gather all supported
    # algorithms (also normalized) as reported by hashlib.
    available = {name.lower() for name in hashlib.algorithms_available}
    algo = hash_algorithm.lower()

    # Explicitly fail fast if the requested algorithm is unknown on this system.
    if algo not in available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # Compute the HMAC. Passing the algorithm name string is supported; hashlib
    # resolves it to the appropriate constructor internally.
    signature = hmac.new(secret, payload, digestmod=algo).digest()

    return signature
