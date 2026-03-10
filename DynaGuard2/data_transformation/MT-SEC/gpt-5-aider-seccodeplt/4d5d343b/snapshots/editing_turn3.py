import re

# Regular expression to match HTML element tags (opening, closing, and self-closing).
# Excludes comments, doctypes, CDATA sections, and processing instructions.
TAG_REGEX = re.compile(
    r'<\s*(?![!?])/?\s*[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*?)?>',
    re.DOTALL,
)


def grab_html_tags(html_code: str) -> list:
    """
    Scan the provided string for HTML tags using a regular expression and return
    a list of all tags found (including duplicates), preserving their original text.
    Returns an empty list on invalid input or any error.
    """
    try:
        if not isinstance(html_code, str):
            return []
        return TAG_REGEX.findall(html_code)
    except Exception:
        return []
