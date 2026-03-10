from typing import Union
import codecs

BytesLike = Union[bytes, bytearray, memoryview]

def validate_byte_content(data: BytesLike) -> bool:
    """
    Validate whether the given bytes-like object is valid UTF-8.

    Args:
        data: A bytes-like object (bytes, bytearray, or memoryview).

    Returns:
        True if the data can be decoded as UTF-8 without errors, otherwise False.
    """
    try:
        # codecs.utf_8_decode accepts any bytes-like object and raises UnicodeDecodeError on invalid data.
        codecs.utf_8_decode(data, errors="strict", final=True)
        return True
    except (UnicodeDecodeError, TypeError):
        return False
