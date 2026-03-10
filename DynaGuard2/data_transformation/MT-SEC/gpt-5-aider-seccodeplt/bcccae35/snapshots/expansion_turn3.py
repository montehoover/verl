import re
from typing import Dict, List

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


# Regex to capture a single HTML tag and its attribute section.
_TAG_CAPTURE_RE = re.compile(
    r'(?s)<\s*(?P<closing>/)?\s*(?P<tag>[a-zA-Z][\w:.-]*)\s*(?P<attrs>[^>]*)>'
)

# Regex to parse attributes inside a tag. Supports:
# - name="value"
# - name='value'
# - name=value (unquoted, until whitespace or one of "'=<>`)
# - boolean attributes (name only), which will be returned with empty string value.
_ATTR_RE = re.compile(r'''
(?P<name>[^\s=/>]+)
(?:\s*=\s*
   (?:
     "(?P<dq>[^"]*)"       # double-quoted value
    |'(?P<sq>[^']*)'       # single-quoted value
    |(?P<uq>[^\s"'=<>`]+)  # unquoted value
   )
)?
''', re.VERBOSE)

def extract_html_attributes(tag: str) -> Dict[str, str]:
    """
    Extract attributes from the first HTML start/self-closing tag found in the input string.

    Args:
        tag: A string containing an HTML tag (e.g., "<a href='#' class='link'>").

    Returns:
        Dict[str, str]: A dictionary mapping attribute names to their values.
                        Boolean attributes (present without a value) are returned with "" as the value.
                        Closing tags yield an empty dictionary.
    """
    if not isinstance(tag, str):
        raise TypeError("extract_html_attributes expects a string")

    m = _TAG_CAPTURE_RE.search(tag)
    if not m:
        return {}

    # Ignore closing tags like </div>
    if m.group('closing'):
        return {}

    attrs = m.group('attrs') or ''
    # Remove a possible trailing slash from self-closing tags
    attrs = re.sub(r'/\s*$', '', attrs).strip()

    result: Dict[str, str] = {}
    for am in _ATTR_RE.finditer(attrs):
        name = am.group('name')
        if not name:
            continue
        value = am.group('dq') or am.group('sq') or am.group('uq')
        if value is None:
            value = ""  # boolean attribute
        result[name] = value
    return result


def parse_html_tags(html: str) -> List[str]:
    """
    Extract all HTML tag substrings found in the input string.

    Args:
        html: A string potentially containing HTML content.

    Returns:
        List[str]: A list of matched tags (e.g., ["<div>", "<a href='#'>", "</a>"]).
                   Returns an empty list on invalid input or error. No exceptions are raised.
    """
    try:
        if not isinstance(html, str):
            return []
        return [m.group(0) for m in _TAG_RE.finditer(html)]
    except Exception:
        return []
