import re

def get_html_tags(html_input: str) -> list:
    """
    Extracts HTML tags from a provided string using a regular expression.

    Args:
        html_input: str, an HTML-formatted string as input.

    Returns:
        A list containing the tags that are present within the input HTML string.
    """
    # Regex to find HTML tags (e.g., <div class="example">, <p>, </span>)
    # It captures the tag name itself.
    # It looks for '<', followed by an optional '/', then the tag name (alphanumeric, can include hyphens),
    # and then either a space, '>', or end of string.
    tag_regex = r"<[/]?([a-zA-Z0-9\-]+)[^>]*>"
    tags = re.findall(tag_regex, html_input)
    return tags
