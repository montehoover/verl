import re
from typing import List

def find_html_tags(html_content: str) -> List[str]:
    """
    Identifies all HTML tags in a given string.

    Args:
        html_content: The string containing HTML content.

    Returns:
        A list of all HTML tags found, including duplicates.
    """
    # Regex to find HTML tags.
    # It looks for an opening '<', followed by an optional '/' or '!',
    # then one or more word characters (for the tag name),
    # then any characters except '>' (for attributes), and finally a closing '>'.
    # This will match entire tags like <p>, <div>, </span>, <!-- comment -->, <!DOCTYPE html>
    tags = re.findall(r"<[/!]?\w[^>]*>", html_content)
    return tags
