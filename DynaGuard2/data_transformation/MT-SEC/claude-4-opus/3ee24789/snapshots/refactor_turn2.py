import re

def extract_html_tags_from_string(html_str: str) -> list:
    """
    Extracts HTML tags from a string using regular expressions.
    
    Args:
        html_str: A string containing HTML content.
    
    Returns:
        A list of HTML tag strings found in the input.
    """
    tag_pattern = r'<[^>]+>'
    return re.findall(tag_pattern, html_str)


def detect_html_tags(html_str: str) -> list:
    """
    Identifies all HTML tags in a given string using regular expressions.
    
    Args:
        html_str: A string representing the input HTML content to be analyzed.
    
    Returns:
        A list containing the strings that match as HTML tags within the input.
    """
    # Delegate the parsing to the pure function
    return extract_html_tags_from_string(html_str)
