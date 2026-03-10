"""
Utilities for preparing passwords for secure storage.
"""

__all__ = ["encode_password"]


def encode_password(password: str) -> bytes:
    """
    Encode the given password string into bytes using UTF-8 encoding.

    Args:
        password: The password as a Unicode string.

    Returns:
        UTF-8 encoded bytes of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8", "strict")
