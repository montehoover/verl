import re

def find_html_tags(html_content: str) -> list[str]:
    """
    Identifies all HTML tags in a given string using regular expressions.

    Args:
        html_content: A string representing the input HTML content to be analyzed.

    Returns:
        A list containing the strings that match as HTML tags within the input.
        Returns an empty list if no tags are found or if input is not a string.
    """

    if not isinstance(html_content, str):
        return []

    # Regular expression to find HTML tags (both opening and closing tags)
    # This pattern matches '<' followed by one or more characters that are not '>', and then '>'
    # It also handles self-closing tags like <img />
    html_tag_pattern = r"<[^>]+>"
    tags = re.findall(html_tag_pattern, html_content)

    return tags
