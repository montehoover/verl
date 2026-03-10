import re

def count_html_tags(html_content: str) -> int:
    """
    Counts the number of HTML tags in a given string.

    Args:
        html_content: The string containing HTML content.

    Returns:
        The count of HTML tags found.
    """
    if not isinstance(html_content, str):
        return 0
    # Regular expression to find HTML tags (simplified version)
    # This regex looks for patterns like <tag_name ...> or </tag_name>
    # It also handles self-closing tags like <tag_name />
    tag_pattern = re.compile(r'<[^>]+>')
    tags = tag_pattern.findall(html_content)
    return len(tags)
