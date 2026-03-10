import re

# Compiled regex to detect HTML tags (opening, closing, and self-closing).
# Matches tags like <div>, </div>, <img />, <a href="...">, including namespaces and hyphens.
_TAG_RE = re.compile(
    r'(?s)<\s*/?\s*[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*?)?\s*/?\s*>'
)

def contains_html(text: str) -> bool:
    """
    Return True if the input string contains any HTML tags, else False.

    Args:
        text: The string to inspect.

    Returns:
        bool: True if HTML tags are present, False otherwise.
    """
    if not isinstance(text, str):
        raise TypeError("contains_html expects a string")
    return bool(_TAG_RE.search(text))
