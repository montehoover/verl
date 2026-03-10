from typing import Final


def extract_printable_content(data: bytes) -> bool:
    """
    Determine whether the given byte sequence contains any readable/printable text.

    Heuristic:
    - If the data decodes as UTF-8, look for a run of at least MIN_RUN_LEN Unicode
      characters where str.isprintable() is True (this excludes control characters like
      newlines and tabs).
    - If UTF-8 decoding fails, fall back to ASCII: look for a run of bytes in the range
      0x20..0x7E (standard printable ASCII, including space).

    Returns True if such a run is found, otherwise False.
    """
    MIN_RUN_LEN: Final[int] = 4

    if not data:
        return False

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # Fallback to ASCII printable range 0x20..0x7E
        run = 0
        for b in data:
            if 0x20 <= b <= 0x7E:
                run += 1
                if run >= MIN_RUN_LEN:
                    return True
            else:
                run = 0
        return False

    # UTF-8 decoded successfully; check for a run of printable Unicode code points
    run = 0
    for ch in text:
        if ch.isprintable():
            run += 1
            if run >= MIN_RUN_LEN:
                return True
        else:
            run = 0
    return False
