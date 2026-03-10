import re
from typing import Pattern, List

# Precompiled regular expression to match HTML tags, excluding comments,
# doctypes, CDATA sections, and processing instructions.
TAG_PATTERN: Pattern[str] = re.compile(
    r'<\s*(?!/?\s*doctype\b)(?!\[CDATA\[)(?!--)(?!\?)[a-z/][^<>]*?>',
    re.IGNORECASE | re.DOTALL,
)

# Pattern to capture the tag name for unique extraction (after cleaning).
TAG_NAME_PATTERN: Pattern[str] = re.compile(
    r'<\s*(?!/?\s*doctype\b)(?!\[CDATA\[)(?!--)(?!\?)/?\s*([a-z][a-z0-9:_-]*)\b[^<>]*?>',
    re.IGNORECASE | re.DOTALL,
)


def count_html_tags(html_content: str) -> int:
    """
    Count the number of HTML tags in the given string using regular expressions.
    Returns 0 for invalid input or on any error. This function avoids raising exceptions.
    """
    try:
        if not isinstance(html_content, str):
            return 0

        content = html_content

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Remove DOCTYPE declarations
        content = re.sub(r'<!DOCTYPE.*?>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # Remove CDATA sections
        content = re.sub(r'<!\[CDATA\[.*?\]\]>', '', content, flags=re.DOTALL)

        # Remove processing instructions (e.g., <?xml ... ?>)
        content = re.sub(r'<\?.*?\?>', '', content, flags=re.DOTALL)

        # Count remaining tags
        return len(TAG_PATTERN.findall(content))
    except Exception:
        return 0


def extract_unique_html_tags(html_content: str) -> List[str]:
    """
    Extract a list of unique HTML tag names from the given string using regular expressions.
    Returns an empty list for invalid input or on any error. This function avoids raising exceptions.
    """
    try:
        if not isinstance(html_content, str):
            return []

        content = html_content

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Remove DOCTYPE declarations
        content = re.sub(r'<!DOCTYPE.*?>', '', content, flags=re.IGNORECASE | re.DOTALL)

        # Remove CDATA sections
        content = re.sub(r'<!\[CDATA\[.*?\]\]>', '', content, flags=re.DOTALL)

        # Remove processing instructions (e.g., <?xml ... ?>)
        content = re.sub(r'<\?.*?\?>', '', content, flags=re.DOTALL)

        # Capture and normalize tag names
        unique_names = set()
        for match in TAG_NAME_PATTERN.finditer(content):
            name = match.group(1).lower()
            unique_names.add(name)

        # Return a deterministic order
        return sorted(unique_names)
    except Exception:
        return []
