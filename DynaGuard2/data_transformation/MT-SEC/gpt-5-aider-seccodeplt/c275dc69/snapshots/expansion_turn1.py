from typing import Union

def validate_byte_stream(data: Union[bytes, bytearray, memoryview]) -> bool:
    """
    Validate that the provided byte stream contains only valid UTF-8 encoded text.

    Args:
        data: A bytes-like object (bytes, bytearray, or memoryview).

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        # Ensure we are working with a bytes object
        b = bytes(data)
        # Attempt strict UTF-8 decoding
        b.decode('utf-8')
        return True
    except (UnicodeDecodeError, TypeError):
        # UnicodeDecodeError -> invalid UTF-8 sequences
        # TypeError -> input was not bytes-like
        return False
