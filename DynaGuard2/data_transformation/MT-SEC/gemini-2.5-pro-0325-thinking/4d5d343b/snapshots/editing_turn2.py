import re
from typing import List

def extract_unique_html_tags(html_content: str) -> List[str]:
    """
    Extracts unique HTML tags from a given string.

    Args:
        html_content: The string containing HTML content.

    Returns:
        A list of unique HTML tags found (e.g., ['html', 'head', 'title', 'body', 'p']).
    """
    if not isinstance(html_content, str):
        return []
    # Regular expression to find HTML tag names.
    # It captures the tag name from opening tags (e.g., <p>, <div class="foo">)
    # and closing tags (e.g., </p>). It ignores self-closing slashes within the tag name part.
    # It extracts the first word character sequence after '<' or '</'.
    tag_pattern = re.compile(r'</?([a-zA-Z0-9]+)(?:\s[^>]*)?>')
    
    # Find all matches. Each match object's group(1) will be the tag name.
    matches = tag_pattern.finditer(html_content)
    
    unique_tags = set()
    for match in matches:
        tag_name = match.group(1).lower() # Normalize to lowercase
        unique_tags.add(tag_name)
        
    return sorted(list(unique_tags))
