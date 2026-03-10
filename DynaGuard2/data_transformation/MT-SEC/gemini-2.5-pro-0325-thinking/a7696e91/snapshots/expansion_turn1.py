import string

def extract_printable_content(data: bytes) -> bool:
    """
    Checks if the given bytes object contains any printable ASCII characters.

    Args:
        data: The bytes object to check.

    Returns:
        True if printable text is found, False otherwise.
    """
    if not isinstance(data, bytes):
        raise TypeError("Input must be a bytes object.")

    # string.printable includes digits, ascii_letters, punctuation, and whitespace.
    # We need to check byte values against their ASCII character representations.
    printable_chars = set(ord(c) for c in string.printable)

    for byte_val in data:
        if byte_val in printable_chars:
            return True
    return False
