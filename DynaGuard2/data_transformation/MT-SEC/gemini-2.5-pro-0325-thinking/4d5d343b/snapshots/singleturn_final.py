import re

def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.
    The function then returns a list containing all detected tags from the input string.

    Args:
        html_code: str, an HTML string that serves as the input for parsing.

    Returns:
        A list of all HTML tags identified in the given input.

    Raises:
        This function should operate without raising exceptions.
    """
    # Regex to find anything between < and >
    # It captures the content inside the tags.
    # For example, in <div class="main">, it captures 'div class="main"'
    # In </p>, it captures '/p'
    tags = re.findall(r'<([^>]+)>', html_code)
    return tags
