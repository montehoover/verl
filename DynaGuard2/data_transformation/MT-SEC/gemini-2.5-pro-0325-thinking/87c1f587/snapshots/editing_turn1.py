import re

def count_html_tags(html_string: str) -> int:
    """
    Counts the number of HTML tags in a given string.

    Args:
        html_string: The string containing HTML content.

    Returns:
        The count of HTML tags found.
    """
    # A simple regex to find HTML tags. It looks for patterns like <tag_name ...> or </tag_name>.
    # This regex might not cover all edge cases of HTML syntax but is a common approach.
    # It specifically looks for an opening '<', followed by an optional '/',
    # then one or more word characters (for the tag name),
    # then any characters except '>' (for attributes), and finally a closing '>'.
    tags = re.findall(r"<[/!]?\w[^>]*>", html_string)
    return len(tags)
