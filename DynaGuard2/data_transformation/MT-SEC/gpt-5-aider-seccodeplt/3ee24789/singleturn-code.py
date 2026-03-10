import re
from typing import List

# Regex to capture the content inside HTML tags, excluding the angle brackets.
# This is a practical approximation and will work for common cases.
_TAG_PATTERN = re.compile(r'<\s*([^<>]+?)\s*>')

def detect_html_tags(html_str: str) -> List[str]:
    """
    Identify all HTML tags in a given string using regular expressions.

    Args:
        html_str (str): A string representing the input HTML content to be analyzed.

    Returns:
        List[str]: A list containing the strings that match as HTML tags within
        the input (contents inside angle brackets, e.g., 'div', '/p', 'a href="#"').

    Notes:
        - This function is resilient: it will not raise exceptions.
        - Declarations, comments, and processing instructions (e.g., <!DOCTYPE ...>,
          <!-- ... -->, <?xml ...?>) are excluded from the results.
    """
    try:
        if not isinstance(html_str, str):
            html_str = '' if html_str is None else str(html_str)

        matches = _TAG_PATTERN.findall(html_str)
        # Normalize and filter out declarations/comments/processing instructions
        tags: List[str] = []
        for m in matches:
            s = m.strip()
            if not s:
                continue
            if s.startswith('!') or s.startswith('?'):
                # Skip declarations, comments, processing instructions
                continue
            tags.append(s)
        return tags
    except Exception:
        # Safety net: never raise
        return []
