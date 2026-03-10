import re

# Precompiled regex to match HTML start, end, and self-closing tags while
# excluding comments, DOCTYPE, and processing instructions.
_TAG_REGEX = re.compile(
    r'<\s*(?![!?])/?\s*[A-Za-z][\w:-]*'
    r'(?:\s+[^\s/>]+(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s\'">=]+))?)*\s*/?>'
)

def count_html_tags(html_content: str) -> int:
    """
    Count the number of HTML tags in the given string.

    Args:
        html_content (str): The HTML content to analyze.

    Returns:
        int: The number of HTML tags found. Returns 0 if input is not a string
             or if any error occurs.
    """
    try:
        if not isinstance(html_content, str):
            return 0
        return len(_TAG_REGEX.findall(html_content))
    except Exception:
        # Ensure no exceptions escape this function
        return 0
