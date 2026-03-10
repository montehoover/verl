"""
SecurePass: utilities for securely hashing passwords.

This module provides a single function, `hash_password`, which hashes a
password string using a specified hash algorithm from Python's hashlib.
"""

import hashlib


# Default digest sizes (in bytes) for variable-length algorithms.
# SHAKE algorithms require an explicit output size; these defaults provide
# broadly compatible digest lengths (256-bit and 512-bit, respectively).
DEFAULT_SHAKE_DIGEST_SIZES = {
    "shake_128": 32,  # 256-bit digest
    "shake_256": 64,  # 512-bit digest
}


def hash_password(hash_name: str, password: str) -> bytes:
    """
    Hash the given password using the specified hash algorithm.

    This function constructs a hashlib object using the provided algorithm
    name, encodes the password as UTF-8, and returns the raw digest bytes.
    For variable-length SHAKE algorithms, it produces a fixed-size digest
    using sensible default lengths.

    Args:
        hash_name: The name of the hash algorithm (e.g., 'sha256',
            'sha3_512', 'shake_128').
        password: The password string to be hashed.

    Returns:
        The hash value of the password as bytes.

    Raises:
        ValueError: If the specified hash algorithm is not available.
    """
    # Normalize the algorithm name to ensure case-insensitive matching.
    algorithm = str(hash_name).lower()

    # Build the hasher. hashlib.new(...) raises ValueError if the algorithm
    # is unknown or not available in the current runtime.
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Hash algorithm not available: {hash_name}") from exc

    # Encode the password as UTF-8 bytes and feed it to the hasher.
    data = password.encode("utf-8")
    hasher.update(data)

    # Handle variable-length algorithms (e.g., SHAKE) which require
    # an explicit digest size. Use module-level defaults to ensure
    # consistent output sizes across calls.
    if hasher.name in DEFAULT_SHAKE_DIGEST_SIZES:
        size = DEFAULT_SHAKE_DIGEST_SIZES[hasher.name]
        return hasher.digest(size)

    # For fixed-length algorithms (e.g., SHA-2, SHA-3, BLAKE2), return
    # the standard digest output.
    return hasher.digest()
