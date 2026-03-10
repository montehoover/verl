import re

def tag_exists(html_string: str, tag: str) -> bool:
    """
    Check if the specified HTML tag exists in the given HTML string.

    Args:
        html_string: The HTML content as a string.
        tag: The tag name to search for (e.g., 'div', 'p').

    Returns:
        True if the tag is found, otherwise False.
    """
    if not isinstance(html_string, str) or not isinstance(tag, str) or not tag:
        return False

    # Matches opening, closing, or self-closing tags, ensuring the tag name is exact.
    pattern = re.compile(
        r'<\s*/?\s*' + re.escape(tag) + r'(?![A-Za-z0-9:-])[^>]*>',
        re.IGNORECASE
    )
    return pattern.search(html_string) is not None
