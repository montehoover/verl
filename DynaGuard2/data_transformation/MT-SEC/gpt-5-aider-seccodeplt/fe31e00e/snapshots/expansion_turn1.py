"""
Utilities for validating byte data encodings.
"""

from typing import ByteString


def validate_byte_data(data: bytes) -> bool:
    """
    Check whether the given bytes are valid UTF-8 encoded data.

    Args:
        data: Byte sequence to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
