import re

def get_html_tags(html_input: str) -> list:
    """
    Extracts HTML tags from a provided string using a regular expression.

    Args:
        html_input: str, an HTML-formatted string as input.

    Returns:
        A list containing the tags that are present within the input HTML string.
        This function should not raise any exceptions under normal operation.
    """
    # Regex to find everything between < and >
    # <         matches the opening angle bracket
    # (.*?)     matches any character (except for line terminators)
    #           *? Quantifier — Matches between zero and unlimited times, as few times as possible, expanding as needed (lazy)
    # >         matches the closing angle bracket
    tags = re.findall(r"<(.*?)>", html_input)
    return tags
