import re

_PATTERN = re.compile(r'^\s*([^(]*)\(([^)]*)\)\s*$')

def match_strings(text: str):
    """
    Match a string of the form 'outside(inside)'.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: (outside, inside) if matched; otherwise None.
    """
    try:
        if not isinstance(text, str):
            return None
        m = _PATTERN.match(text)
        if not m:
            return None
        return m.group(1), m.group(2)
    except Exception:
        return None
