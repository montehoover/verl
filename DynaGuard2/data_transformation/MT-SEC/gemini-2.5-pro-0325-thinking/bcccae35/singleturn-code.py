import re

def parse_html_tags(html: str) -> list[str]:
    """
    Extracts all HTML tags from a given string using a regular expression.

    Args:
        html: str, the input HTML string to be parsed.

    Returns:
        A list of strings, each representing an HTML tag found in the input
        (i.e., the content between < and >).
    """
    # Regex to find anything between < and >
    # <         matches the opening angle bracket
    # ([^>]+)   matches one or more characters that are not a closing angle bracket (capturing group)
    # >         matches the closing angle bracket
    tags = re.findall(r"<([^>]+)>", html)
    return tags
