import re

_URL_PATTERN = re.compile(
    r'(?i)\b(?:'
    r'(?:https?://|ftp://)'  # protocol
    r'|www\d{0,3}\.'         # or www.
    r')'
    r'[^\s<>\(\)\[\]{}"\'`]+'
)

def detect_html_tags(content: str) -> list[str]:
    """
    Detects all URLs in the given content string using regular expressions.

    Args:
        content: The input text to analyze.

    Returns:
        A list of strings where each string is a detected URL.
    """
    if not isinstance(content, str) or not content:
        return []

    matches = _URL_PATTERN.findall(content)

    # Remove common trailing punctuation that may be attached to the URL.
    cleaned = [re.sub(r'[.,;:!?)"\]>]+$', '', m) for m in matches]

    return cleaned
