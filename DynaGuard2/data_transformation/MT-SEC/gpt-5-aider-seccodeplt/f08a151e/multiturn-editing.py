import re

_HTML_TAG_PATTERN = re.compile(
    r'</?\s*[A-Za-z][A-Za-z0-9:_-]*'                       # tag name (start or end)
    r'(?:\s+(?:[^\s"\'<>/=]+'                              # attribute name
    r'(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s"\'=<>`]+))?'    # optional = value
    r'))*\s*/?>',                                          # optional self-closing
    re.IGNORECASE
)

def get_html_tags(html_input: str):
    """
    Extract all HTML tags from the given HTML-formatted string using regular expressions.

    Args:
        html_input (str): Input HTML string.

    Returns:
        list[str]: List of HTML tag strings found in the input, in scan order.
    """
    try:
        if not isinstance(html_input, str) or not html_input:
            return []
        return [m.group(0) for m in _HTML_TAG_PATTERN.finditer(html_input)]
    except Exception:
        return []
