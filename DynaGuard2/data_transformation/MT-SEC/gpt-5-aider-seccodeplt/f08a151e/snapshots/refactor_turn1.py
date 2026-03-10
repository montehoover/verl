import re

# Pre-compiled regex to match start tags, end tags, and self-closing tags.
# Excludes comments (<!-- -->), doctypes (<!DOCTYPE ...>), and processing instructions (<? ... ?>).
_TAG_REGEX = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?>")

def get_html_tags(html_input: str) -> list:
    """
    Extract HTML tags from a provided string using a regular expression.

    Args:
        html_input: An HTML-formatted string.

    Returns:
        A list of matched HTML tags (as strings) found in the input.
    """
    try:
        if not isinstance(html_input, str):
            html_input = str(html_input)
        if not html_input:
            return []
        return _TAG_REGEX.findall(html_input)
    except Exception:
        # Do not raise exceptions under normal operation; return empty list on unexpected errors.
        return []
