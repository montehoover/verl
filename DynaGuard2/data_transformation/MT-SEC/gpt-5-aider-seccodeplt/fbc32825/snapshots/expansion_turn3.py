from typing import List, Dict, Optional, Tuple
import re

def split_text(text: str, delimiter: str) -> List[str]:
    """
    Splits the given text by the specified delimiter and returns a list of substrings.

    Args:
        text: The input string to split.
        delimiter: The substring delimiter to split on.

    Returns:
        A list of substrings.
    """
    return text.split(delimiter)

def identify_parts(text: str) -> Dict[str, str]:
    """
    Identifies and separates parts of a string formatted as 'prefix(suffix)'.

    Returns a dictionary with:
    - prefix: The section of the string before the parentheses
    - suffix: The section of the string inside the parentheses
    """
    open_idx = text.find('(')
    close_idx = text.rfind(')')

    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        raise ValueError("Input must be formatted as 'prefix(suffix)'")

    prefix = text[:open_idx]
    suffix = text[open_idx + 1:close_idx]
    return {"prefix": prefix, "suffix": suffix}

def match_strings(text: str) -> Optional[Tuple[str, str]]:
    """
    Uses a regular expression to match a string formatted as 'outside(inside)'.

    Returns:
        A tuple (outside, inside) if matched, or None if the pattern does not match.
        This function does not raise exceptions.
    """
    try:
        if not isinstance(text, str):
            return None
        match = re.match(r'^(?P<outside>[^()]*)\((?P<inside>[^()]*)\)$', text)
        if not match:
            return None
        return match.group('outside'), match.group('inside')
    except Exception:
        return None
