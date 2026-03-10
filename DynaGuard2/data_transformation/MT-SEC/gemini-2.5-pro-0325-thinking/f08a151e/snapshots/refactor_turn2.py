import re

def get_html_tags(html_input: str) -> list:
    """
    Extracts HTML tags from a provided string using a regular expression.

    Args:
        html_input: str, an HTML-formatted string as input.

    Returns:
        A list containing the tags that are present within the input HTML string.
    """
    # Regex to find HTML tags (e.g., <div class="example">, <p>, </span>).
    # This pattern captures the tag name.
    # It looks for:
    #   '<' followed by an optional '/' (for closing tags),
    #   then the tag name (alphanumeric characters and hyphens),
    #   followed by any characters except '>' until the closing '>'.
    html_tag_pattern = r"<[/]?([a-zA-Z0-9\-]+)[^>]*>"
    extracted_tags = re.findall(html_tag_pattern, html_input)
    
    return extracted_tags
