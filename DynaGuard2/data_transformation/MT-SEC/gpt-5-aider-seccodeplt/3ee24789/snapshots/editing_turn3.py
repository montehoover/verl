import re

_HTML_TAG_PATTERN = re.compile(
    r'(?is)<\s*/?\s*[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+(?:"[^"]*"|\'[^\']*\'|[^\'">]))*\s*/?\s*>'
)

def detect_html_tags(html_str: str) -> list[str]:
    """
    Identify all HTML tags within the given string.

    Args:
        html_str: The input HTML content to analyze.

    Returns:
        A list of strings where each string is a detected HTML tag (e.g., '<div>', '<p>', '<span class="x">').
    """
    if not isinstance(html_str, str) or not html_str:
        return []

    return [m.group(0) for m in _HTML_TAG_PATTERN.finditer(html_str)]
