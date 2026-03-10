import re


def parse_html_tags(html: str) -> list[str]:
    """
    Extract all HTML tags from a given string using regular expression.
    
    This function searches for HTML tags in the input string and returns
    a list of all tags found, including opening tags, closing tags, and
    self-closing tags.
    
    Args:
        html (str): The input HTML string to be parsed.
        
    Returns:
        list[str]: A list of strings, each representing an HTML tag found
                   in the input.
    """
    # Pattern explanation:
    # < - matches the opening angle bracket
    # [^>]+ - matches one or more characters that are not closing angle brackets
    # > - matches the closing angle bracket
    pattern = r'<[^>]+>'
    
    return re.findall(pattern, html)
