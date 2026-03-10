def tag_exists(html_string: str, tag: str) -> bool:
    """
    Checks if a specific HTML tag is present in a given string.
    This is a simple check and looks for "<tag_name" (for opening tags like <tag> or <tag attr>)
    or "</tag_name>" (for closing tags like </tag>).
    The check is case-sensitive.

    Args:
        html_string: The string containing HTML content.
        tag: The HTML tag name to search for (e.g., "p", "div").

    Returns:
        True if the tag is found, False otherwise.
    """
    opening_tag_marker = f"<{tag}"  # Matches "<tag" which covers <tag> and <tag ...>
    closing_tag_marker = f"</{tag}>" # Matches "</tag>"

    if opening_tag_marker in html_string or closing_tag_marker in html_string:
        return True
    return False
