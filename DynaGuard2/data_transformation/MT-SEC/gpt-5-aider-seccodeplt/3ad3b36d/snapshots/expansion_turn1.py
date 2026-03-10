"""
Utilities for password processing.

This module provides foundational helpers for preparing plaintext passwords
before hashing or storage.
"""

from typing import Final

__all__: Final = ["encode_password"]


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
