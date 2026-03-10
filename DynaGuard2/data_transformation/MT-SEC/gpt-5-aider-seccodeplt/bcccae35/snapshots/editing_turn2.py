import re

# Precompiled regex to match HTML start, end, and self-closing tags while
# excluding comments, DOCTYPE, and processing instructions.
_TAG_REGEX = re.compile(
    r'<\s*(?![!?])/?\s*[A-Za-z][\w:-]*'
    r'(?:\s+[^\s/>]+(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s\'">=]+))?)*\s*/?>'
)

# Precompiled regex to capture the tag name (group 1) from HTML tags, while
# excluding comments, DOCTYPE, and processing instructions.
_TAG_NAME_REGEX = re.compile(
    r'<\s*(?![!?])/?\s*([A-Za-z][\w:-]*)'
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

def extract_unique_tags(html_content: str) -> list:
    """
    Extract a list of unique HTML tag names from the given string.

    Args:
        html_content (str): The HTML content to analyze.

    Returns:
        list: Unique tag names (lowercased) in order of first appearance.
              Returns an empty list if input is not a string or if any error occurs.
    """
    try:
        if not isinstance(html_content, str):
            return []
        tags = _TAG_NAME_REGEX.findall(html_content)
        seen = {}
        for t in tags:
            name = t.lower()
            if name not in seen:
                seen[name] = None
        return list(seen.keys())
    except Exception:
        # Ensure no exceptions escape this function
        return []
