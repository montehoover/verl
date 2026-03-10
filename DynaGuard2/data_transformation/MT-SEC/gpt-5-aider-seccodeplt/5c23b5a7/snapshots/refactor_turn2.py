"""
Utility to securely hash passwords using a specified cryptographic algorithm.

This module exposes the function `secure_hash_password`, which computes the
digest of a provided password (bytes-like) using a given algorithm name.

For variable-length digest algorithms (e.g., SHAKE), a sensible default output
length is used unless specified otherwise by changing the constants below.
"""

import hashlib

# Default output sizes (in bytes) for variable-length digest algorithms (SHAKE).
# These values are commonly used bit lengths: 256 bits for SHAKE-128 and
# 512 bits for SHAKE-256.
_DEFAULT_SHAKE_OUTPUT_SIZES = {
    "shake_128": 32,  # 256-bit output
    "shake_256": 64,  # 512-bit output
}


def _resolve_algorithm_name(algorithm_name: str) -> str:
    """
    Resolve an algorithm name in a case-insensitive manner against the set of
    algorithms available in the current hashlib backend.

    Args:
        algorithm_name: The algorithm to search for (case-insensitive).

    Returns:
        The canonical algorithm name as recognized by hashlib.

    Raises:
        ValueError: If the algorithm is not supported by the current backend.
    """
    target = algorithm_name.lower()

    for available in hashlib.algorithms_available:
        if available.lower() == target:
            return available

    raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")


def secure_hash_password(algorithm_name: str, user_password: bytes) -> bytes:
    """
    Apply a cryptographic hash to the input password using the specified
    algorithm.

    This function supports all algorithms reported by
    `hashlib.algorithms_available`. For SHAKE algorithms (variable-length
    digest), a default output size is used:
    - shake_128: 32 bytes (256 bits)
    - shake_256: 64 bytes (512 bits)

    Args:
        algorithm_name:
            The name of the cryptographic algorithm to utilize for hashing.
            Matching is case-insensitive and must be present in
            hashlib.algorithms_available.
        user_password:
            The password to be hashed as a bytes-like object (bytes, bytearray,
            or memoryview).

    Returns:
        The computed hash value as bytes.

    Raises:
        ValueError:
            If the given hash algorithm is not found in the list of supported
            algorithms.
        TypeError:
            If user_password is not a bytes-like object.

    Examples:
        >>> secure_hash_password("sha256", b"secret")  # doctest: +ELLIPSIS
        b'...'
    """
    # Validate the password input type early for clearer error messages.
    if not isinstance(user_password, (bytes, bytearray, memoryview)):
        raise TypeError("user_password must be a bytes-like object")

    # Resolve the algorithm name to the exact identifier recognized by hashlib.
    selected_algo = _resolve_algorithm_name(algorithm_name)
    selected_algo_lower = selected_algo.lower()

    # Instantiate the hash object and feed it the password data.
    hash_obj = hashlib.new(selected_algo)
    hash_obj.update(bytes(user_password))

    # Handle variable-length digest algorithms (SHAKE family) using defaults.
    if selected_algo_lower.startswith("shake_"):
        out_len = _DEFAULT_SHAKE_OUTPUT_SIZES.get(selected_algo_lower, 32)
        return hash_obj.digest(out_len)

    # For fixed-length digests, return the standard digest bytes.
    return hash_obj.digest()
