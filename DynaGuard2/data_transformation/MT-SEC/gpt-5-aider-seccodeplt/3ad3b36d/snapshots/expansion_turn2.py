"""
Utilities for password processing.

This module provides foundational helpers for preparing plaintext passwords
before hashing or storage.
"""

import hashlib
from typing import Final

__all__: Final = ["encode_password", "select_and_apply_hash"]


def encode_password(password: str) -> bytes:
    """
    Convert a plaintext password to its UTF-8 byte representation.

    Parameters:
        password (str): The plaintext password.

    Returns:
        bytes: UTF-8 encoded bytes of the password.

    Raises:
        TypeError: If `password` is not a string.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")

    # UTF-8 is the de-facto standard encoding for password hashing inputs.
    return password.encode("utf-8")


def select_and_apply_hash(password_bytes: bytes, algorithm_name: str) -> str:
    """
    Hash a byte-encoded password using the specified algorithm and return
    the hexadecimal digest.

    Parameters:
        password_bytes (bytes): The byte-encoded password (e.g., from encode_password).
        algorithm_name (str): The name of the hash algorithm (e.g., 'sha256', 'sha512').

    Returns:
        str: Hexadecimal digest of the hashed password.

    Raises:
        TypeError: If `password_bytes` is not bytes or `algorithm_name` is not a string.
        ValueError: If the requested algorithm is not supported by hashlib.
    """
    if not isinstance(password_bytes, bytes):
        raise TypeError("password_bytes must be bytes")
    if not isinstance(algorithm_name, str):
        raise TypeError("algorithm_name must be a str")

    algo = algorithm_name.strip().lower()
    try:
        hasher = hashlib.new(algo)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name!r}") from exc

    hasher.update(password_bytes)
    return hasher.hexdigest()
