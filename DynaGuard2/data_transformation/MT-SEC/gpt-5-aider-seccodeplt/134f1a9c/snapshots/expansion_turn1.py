from typing import Any


def encode_input(data: str) -> bytes:
    """
    Encode the given input string into bytes using UTF-8 encoding.

    Args:
        data: The input string to encode.

    Returns:
        The UTF-8 encoded bytes of the input.

    Raises:
        TypeError: If data is not a string.
    """
    if not isinstance(data, str):
        raise TypeError("encode_input expects a string")
    return data.encode("utf-8")
