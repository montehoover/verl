import re
from typing import List

def list_unique_html_tags(html_string: str) -> List[str]:
    """
    Lists the unique HTML tags found in a given string.

    Args:
        html_string: The string containing HTML content.

    Returns:
        A list of unique HTML tags found.
    """
    # Regex to find HTML tag names.
    # It looks for an opening '<', optionally followed by '/' (for closing tags) or '!' (for doctype),
    # then captures one or more word characters (the tag name),
    # followed by any characters except '>' (for attributes or self-closing slash),
    # and finally a closing '>'.
    # Example: <div class="foo"> -> "div", </body> -> "body", <br/> -> "br"
    tag_names = re.findall(r"<[/!]?(\w+)[^>]*>", html_string)
    
    # Using a set to store unique tag names and then converting to a list
    unique_tags = sorted(list(set(tag_name.lower() for tag_name in tag_names)))
    return unique_tags
