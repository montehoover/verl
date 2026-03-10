import re

# Compiled regular expression to match HTML tags (opening, self-closing, and closing),
# while excluding comments, DOCTYPE, and processing/server instructions.
_TAG_RE = re.compile(
    r"(?:"
    r"</\s*[a-zA-Z][a-zA-Z0-9:-]*\s*>"  # closing tags, e.g., </div>
    r"|"
    r"<\s*(?!/?(?:!--|!doctype|\?|%))"  # exclude comments, doctype, PIs, server tags
    r"[a-zA-Z][a-zA-Z0-9:-]*"           # tag name
    r"(?:\s+[a-zA-Z_:][a-zA-Z0-9:._-]*" # attribute name
    r"(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))?)*"  # attribute value
    r"\s*/?>"                           # end of tag
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Compiled regex to capture tag names from both opening/self-closing and closing tags,
# excluding comments, DOCTYPE, and processing/server instructions.
_TAG_NAME_RE = re.compile(
    r"</\s*([a-zA-Z][a-zA-Z0-9:-]*)\s*>"
    r"|"
    r"<\s*(?!/?(?:!--|!doctype|\?|%))([a-zA-Z][a-zA-Z0-9:-]*)\b[^>]*?>",
    re.IGNORECASE | re.DOTALL,
)

def count_html_tags(html_string: str) -> int:
    """
    Count the number of HTML tags in the given string using a regular expression.
    The function is defensive and will not raise exceptions; it returns 0 on error.

    Args:
        html_string: The HTML content as a string.

    Returns:
        The count of HTML tags found.
    """
    try:
        if not isinstance(html_string, str):
            return 0
        matches = _TAG_RE.findall(html_string)
        return len(matches)
    except Exception:
        return 0

def list_unique_html_tags(html_string: str) -> list:
    """
    Return a list of unique HTML tag names present in the given string.
    Tag names are normalized to lowercase. On error or non-string input, returns [].
    """
    try:
        if not isinstance(html_string, str):
            return []
        tags = set()
        for close_name, open_name in _TAG_NAME_RE.findall(html_string):
            name = close_name or open_name
            if name:
                tags.add(name.lower())
        return sorted(tags)
    except Exception:
        return []
