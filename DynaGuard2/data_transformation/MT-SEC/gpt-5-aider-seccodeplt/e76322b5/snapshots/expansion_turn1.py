def is_ascii_printable(byte_stream):
    """
    Check whether the given byte_stream contains only ASCII printable characters.

    ASCII printable range: 0x20 (space) to 0x7E (~), inclusive.
    Returns:
        bool: True if all bytes are in the printable ASCII range, False otherwise.
    """
    try:
        data = bytes(byte_stream)
    except Exception:
        return False

    return all(32 <= b <= 126 for b in data)
