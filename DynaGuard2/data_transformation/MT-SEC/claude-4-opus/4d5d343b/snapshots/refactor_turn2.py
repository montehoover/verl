import re


def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.
    
    This function parses an HTML string and extracts all HTML tags,
    including opening tags (e.g., <div>), closing tags (e.g., </div>),
    and self-closing tags (e.g., <br />).
    
    Args:
        html_code (str): An HTML string that serves as the input for parsing.
    
    Returns:
        list: A list of all HTML tags identified in the given input.
              Each tag is returned as a string including the angle brackets.
    
    Example:
        >>> grab_html_tags("<p>Hello <b>world</b></p>")
        ['<p>', '<b>', '</b>', '</p>']
    """
    # Regular expression pattern to match HTML tags
    # Pattern explanation:
    # < : Matches the opening angle bracket
    # [^>]+ : Matches one or more characters that are NOT a closing angle bracket
    # > : Matches the closing angle bracket
    # This pattern captures any content between < and >, including tag names,
    # attributes, and forward slashes for closing tags
    pattern = r'<[^>]+>'
    
    # Find all matches in the HTML code using the compiled pattern
    tags = re.findall(pattern, html_code)
    
    return tags
