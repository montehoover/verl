from typing import Any

__all__ = ["deserialize_data"]

def deserialize_data(data: Any, encoding: str = "utf-8", errors: str = "strict") -> str:
    """
    Convert a byte-like data stream into a string using simple decoding.

    Parameters:
        data: A bytes-like object (e.g., bytes, bytearray, memoryview) or str.
        encoding: Text encoding to use (default: 'utf-8').
        errors: Error handling scheme (default: 'strict').

    Returns:
        The decoded string.

    Raises:
        TypeError: If 'data' is not bytes-like or str.
        UnicodeDecodeError: If decoding fails and errors='strict'.
    """
    if isinstance(data, str):
        return data

    try:
        b = bytes(data)
    except TypeError:
        raise TypeError("deserialize_data expects a bytes-like object or str") from None

    return b.decode(encoding, errors=errors)
