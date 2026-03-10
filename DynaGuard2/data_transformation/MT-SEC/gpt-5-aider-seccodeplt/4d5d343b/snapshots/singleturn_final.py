import re

# Pre-compiled regex to find HTML tags and capture their inner content (without < and >).
# Excludes comments/DOCTYPE/processing instructions (i.e., tags starting with '!' or '?').
_TAG_PATTERN = re.compile(r"<\s*(?![!?])([^<>]+?)\s*>")

def grab_html_tags(html_code: str):
    """
    Scans the provided string for HTML tags using a regular expression and returns a list
    containing the content of all detected tags (excluding angle brackets).

    Args:
        html_code (str): An HTML string that serves as the input for parsing.

    Returns:
        list: A list of all HTML tags identified in the given input, with each entry being the
              inner content of the tag (e.g., 'div', '/p', "a href='#'", 'img src="x"/').

    Notes:
        - This function is designed not to raise exceptions; it returns an empty list on invalid input
          or unexpected errors.
        - It excludes matches for comments, DOCTYPE, and processing instructions (<!...>, <?...>).
    """
    try:
        if not isinstance(html_code, str):
            return []
        return [m.group(1).strip() for m in _TAG_PATTERN.finditer(html_code)]
    except Exception:
        # Fail-safe: never raise from this function
        return []
