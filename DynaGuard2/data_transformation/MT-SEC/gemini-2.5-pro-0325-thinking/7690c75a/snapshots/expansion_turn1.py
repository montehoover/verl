from typing import Union

def validate_byte_stream(stream: Union[bytes, bytearray]) -> bool:
    """
    Validates if the given byte stream is non-empty and not corrupted.

    Args:
        stream: The byte stream to validate. It should be a bytes or bytearray object.

    Returns:
        True if the stream is a non-empty bytes or bytearray object, False otherwise.
    """
    if not isinstance(stream, (bytes, bytearray)):
        # Not a valid byte stream type
        return False

    if not stream:
        # Empty stream
        return False

    # At this point, the stream is a non-empty bytes or bytearray object.
    # More sophisticated "corruption" checks (e.g., checksum, magic numbers)
    # would depend on the specific format of the byte stream and are not
    # implemented in this generic function.
    return True
