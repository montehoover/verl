"""
Utilities for hashing passwords using Python's hashlib.

This module provides a single public function, `hash_password`, which hashes a
plaintext password using the specified algorithm and returns its hexadecimal
representation. It leverages hashlib and supports any algorithm reported by
`hashlib.algorithms_available`. For SHAKE algorithms (shake_128 and shake_256),
which are extensible-output functions, sensible default output lengths are used.
"""

import hashlib


# Default output lengths (in bytes) for SHAKE algorithms.
# SHAKE functions require an explicit digest size; these defaults provide
# commonly expected security levels:
# - shake_128 -> 32 bytes (256-bit)
# - shake_256 -> 64 bytes (512-bit)
DEFAULT_SHAKE_LENGTHS = {
    "shake_128": 32,
    "shake_256": 64,
}


def _normalize_algorithm_name(algo_name: str) -> str:
    """
    Normalize and validate the provided algorithm name.

    Args:
        algo_name: The raw algorithm name input.

    Returns:
        A lowercase, trimmed algorithm name.

    Raises:
        ValueError: If the provided value is not a string or is empty after trimming.
    """
    if not isinstance(algo_name, str):
        raise ValueError(
            "Algorithm name must be a string representing a supported hash algorithm"
        )

    algo = algo_name.strip().lower()
    if not algo:
        raise ValueError("Algorithm name cannot be empty")

    return algo


def _validate_algorithm_available(algo: str) -> None:
    """
    Validate that the algorithm is supported by the current OpenSSL/hashlib build.

    Args:
        algo: Normalized algorithm name.

    Raises:
        ValueError: If the algorithm is not available.
    """
    available = {a.lower() for a in hashlib.algorithms_available}
    if algo not in available:
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo}")


def hash_password(algo_name: str, secret: str) -> str:
    """
    Hash a plaintext password using the specified hash algorithm and return its hex digest.

    Args:
        algo_name: The name of the hash algorithm to use (e.g., 'sha256', 'blake2b', 'shake_128').
        secret: The plaintext password to be hashed.

    Returns:
        A string containing the hexadecimal representation of the hashed password.

    Raises:
        ValueError: If an unsupported or unavailable hash algorithm is provided.

    Notes:
        - For SHAKE algorithms (shake_128, shake_256), this function uses default
          output lengths defined in DEFAULT_SHAKE_LENGTHS.
        - The `secret` input is encoded using UTF-8 prior to hashing.
    """
    # Normalize and validate the algorithm name.
    algo = _normalize_algorithm_name(algo_name)
    _validate_algorithm_available(algo)

    # Ensure the secret is a string; convert if necessary for robustness.
    if not isinstance(secret, str):
        secret = str(secret)

    # Create a new hash object for the requested algorithm.
    hash_obj = hashlib.new(algo)

    # Feed the plaintext bytes (UTF-8 encoded) into the hash object.
    hash_obj.update(secret.encode("utf-8"))

    # SHAKE algorithms require an explicit digest size for hexdigest().
    if algo in DEFAULT_SHAKE_LENGTHS:
        return hash_obj.hexdigest(DEFAULT_SHAKE_LENGTHS[algo])

    # All other algorithms expose a fixed-length digest.
    return hash_obj.hexdigest()
