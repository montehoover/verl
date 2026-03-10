import re


def extract_html_tags_from_string(html_str: str) -> list:
    """
    Extracts HTML tags from a string using regular expressions.
    
    This function uses a regular expression pattern to find all HTML tags
    within the provided string, including opening tags, closing tags,
    and self-closing tags.
    
    Args:
        html_str: A string containing HTML content to be parsed.
    
    Returns:
        list: A list of strings, where each string is an HTML tag found
              in the input. Tags are returned in the order they appear
              in the original string.
    
    Example:
        >>> extract_html_tags_from_string('<div>Hello <b>world</b></div>')
        ['<div>', '<b>', '</b>', '</div>']
    """
    tag_pattern = r'<[^>]+>'
    return re.findall(tag_pattern, html_str)


def detect_html_tags(html_str: str) -> list:
    """
    Identifies all HTML tags in a given string using regular expressions.
    
    This is the main entry point for HTML tag detection. It delegates
    the actual parsing work to a specialized function for better
    maintainability and separation of concerns.
    
    Args:
        html_str: A string representing the input HTML content to be analyzed.
    
    Returns:
        list: A list containing the strings that match as HTML tags within
              the input. Each tag is returned as a complete string including
              the angle brackets.
    
    Note:
        This function does not validate whether the tags are valid HTML tags,
        it simply extracts anything that matches the pattern of an HTML tag.
    """
    # Delegate the parsing to the pure function
    return extract_html_tags_from_string(html_str)
