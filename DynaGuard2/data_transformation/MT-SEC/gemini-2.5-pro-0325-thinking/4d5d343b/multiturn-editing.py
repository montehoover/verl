import re
from typing import List

def grab_html_tags(html_code: str) -> List[str]:
    """
    Scans a provided string for HTML tags using a regular expression.

    Args:
        html_code: The string containing HTML code.

    Returns:
        A list of all HTML tags identified in the given input.
    """
    if not isinstance(html_code, str):
        return []
    # Regular expression to find HTML tags (e.g., <p>, </p>, <br />, <img src="...">)
    tag_pattern = re.compile(r'<[^>]+>')
    tags = tag_pattern.findall(html_code)
    return tags
