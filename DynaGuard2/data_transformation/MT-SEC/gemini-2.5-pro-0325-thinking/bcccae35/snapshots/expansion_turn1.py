import re

def contains_html(input_string: str) -> bool:
    """
    Checks if a string contains any HTML tags.

    Args:
        input_string: The string to check.

    Returns:
        True if HTML tags are present, False otherwise.
    """
    # A simple regex to find anything that looks like an HTML tag
    # This will match <tag>, <tag arg="value">, </tag>, <!-- comment -->, etc.
    html_pattern = re.compile(r'<[^>]+>')
    return bool(html_pattern.search(input_string))
