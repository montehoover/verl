import re

_URL_PATTERN = re.compile(
    r'(?i)\b(?:'
    r'(?:https?://|ftp://)'  # protocol
    r'|www\d{0,3}\.'         # or www.
    r')'
    r'[^\s<>\(\)\[\]{}"\'`]+'
)

_EMAIL_PATTERN = re.compile(
    r'(?i)\b[0-9a-z._%+-]+@[0-9a-z.-]+\.[a-z]{2,}\b'
)

def detect_html_tags(content: str) -> list[str]:
    """
    Detects all URLs and email addresses in the given content string using regular expressions.

    Args:
        content: The input text to analyze.

    Returns:
        A list of strings where each string is a detected URL or email address.
    """
    if not isinstance(content, str) or not content:
        return []

    tokens: list[tuple[int, str]] = []

    for m in _URL_PATTERN.finditer(content):
        tokens.append((m.start(), m.group(0)))

    for m in _EMAIL_PATTERN.finditer(content):
        tokens.append((m.start(), m.group(0)))

    tokens.sort(key=lambda x: x[0])

    # Remove common trailing punctuation that may be attached to a match (especially URLs).
    cleaned = [re.sub(r'[.,;:!?)"\]>]+$', '', value) for _, value in tokens]

    return cleaned
