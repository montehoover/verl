import re

_PATTERN = re.compile(r'^\s*([^(]*)\(([^)]*)\)\s*$')


def _extract_outside_inside(text: str):
    """
    Pure helper function that performs the regex-based extraction.

    This function attempts to match the input against the pattern:
    'outside(inside)' (allowing leading/trailing whitespace). If matched,
    it returns a tuple of (outside, inside); otherwise, it returns None.

    Notes:
        - Assumes 'text' is a string. Type validation and exception handling
          are delegated to the caller.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: (outside, inside) if matched; otherwise None.
    """
    m = _PATTERN.match(text)
    if not m:
        return None
    return m.group(1), m.group(2)


def match_strings(text: str):
    """
    Match a string of the form 'outside(inside)' and return the captured parts.

    This is the public API that focuses on input/output flow and delegates the
    regex matching to a dedicated helper function for better readability and
    maintainability.

    Behavior:
        - Returns (outside, inside) when the input matches the pattern.
        - Returns None if the input is not a string, does not match,
          or if any unexpected error occurs.
        - Does not raise exceptions.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: (outside, inside) if matched; otherwise None.
    """
    try:
        if not isinstance(text, str):
            return None
        return _extract_outside_inside(text)
    except Exception:
        return None
