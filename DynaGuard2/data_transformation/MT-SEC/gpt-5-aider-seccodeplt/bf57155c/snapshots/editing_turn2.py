import re
from typing import Optional

_TAG_RE = re.compile(
    r'<\s*/?\s*'
    r'[A-Za-z][A-Za-z0-9:_-]*'
    r'(?:'
        r'(?:\s+'
            r'[A-Za-z_:][-\w:.]*'
            r'(?:\s*=\s*'
                r'(?:'
                    r'"[^"]*"'
                    r"|\'[^\']*\'"
                    r'|[^\'"\s<>]+'
                r')'
            r')?'
        r')*'
    r')'
    r'\s*/?\s*>',
    re.DOTALL
)

def contains_html_tags(text: str) -> bool:
    """
    Return True if the given text contains any HTML tags; otherwise False.
    Uses a regular expression and never raises exceptions.
    """
    try:
        if not isinstance(text, str):
            return False
        return _TAG_RE.search(text) is not None
    except Exception:
        return False

def extract_first_html_tag(text: str) -> Optional[str]:
    """
    Return the first HTML tag found in the given text as a string.
    Returns None if no tags are present. Uses a regular expression and never raises exceptions.
    """
    try:
        if not isinstance(text, str):
            return None
        match = _TAG_RE.search(text)
        return match.group(0) if match else None
    except Exception:
        return None
