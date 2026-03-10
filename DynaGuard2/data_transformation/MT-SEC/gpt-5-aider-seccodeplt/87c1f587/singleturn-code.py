import re

# Precompiled pattern to capture the inner content of HTML tags (excluding angle brackets).
# - Captures opening/closing tags and any attributes (e.g., "a href='#'", "/div").
# - Ignores comments (<!-- ... -->), declarations (<!DOCTYPE ...>), and processing instructions (<? ... ?>).
_TAG_PATTERN = re.compile(r'<\s*(?!(?:!--|!|\?))([^>]+?)\s*>')

def find_html_tags(html_content: str):
    """
    Identifies all HTML tags in a given string using regular expressions.

    Args:
        html_content (str): A string representing the input HTML content to be analyzed.

    Returns:
        list[str]: A list containing the strings that match as HTML tags within the input.

    Notes:
        - This function is resilient and will not raise exceptions. If an error occurs,
          it returns an empty list.
        - It ignores HTML comments, declarations (e.g., <!DOCTYPE ...>), and processing
          instructions (e.g., <?xml ... ?>).
    """
    try:
        if not isinstance(html_content, str):
            html_content = "" if html_content is None else str(html_content)
        return _TAG_PATTERN.findall(html_content)
    except Exception:
        # As required, do not raise; return an empty list on any unexpected error.
        return []
