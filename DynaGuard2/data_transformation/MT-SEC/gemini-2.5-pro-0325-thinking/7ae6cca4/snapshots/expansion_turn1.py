import string

def is_safe_byte_content(data: bytes) -> bool:
    """
    Checks if a byte sequence contains only secure printable ASCII characters.

    Secure printable characters are defined as those in the ASCII range 32 (space)
    to 126 (~), inclusive.

    Args:
        data: The byte sequence to check.

    Returns:
        True if all bytes in the sequence are secure printable ASCII characters,
        False otherwise.

    Raises:
        TypeError: If the input is not a bytes object.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")

    for byte_val in data:
        # ASCII printable characters are in the range 32 (space) to 126 (~)
        if not (32 <= byte_val <= 126):
            return False
    return True
