__all__ = ["is_safe_byte_content"]


def is_safe_byte_content(data: bytes) -> bool:
    """
    Return True if 'data' contains only secure printable characters, False otherwise.

    Allowed bytes:
    - ASCII printable characters 0x20 (space) through 0x7E (~)
    - Horizontal tab (0x09), line feed (0x0A), carriage return (0x0D)
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("is_safe_byte_content expects a bytes-like object")

    for b in data:
        if not (b == 0x09 or b == 0x0A or b == 0x0D or 0x20 <= b <= 0x7E):
            return False
    return True
