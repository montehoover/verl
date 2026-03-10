import re


def identify_html_tags(content: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.
    
    This function identifies all HTML tags within the given content string,
    including opening tags (e.g., <div>), closing tags (e.g., </div>),
    and self-closing tags (e.g., <br />).
    
    Args:
        content (str): An HTML string that serves as the input for parsing.
        
    Returns:
        list: A list of all HTML tags identified in the given input.
              Each tag is returned as a string including the angle brackets.
              
    Example:
        >>> html_content = '<div>Hello <b>world</b>!</div>'
        >>> identify_html_tags(html_content)
        ['<div>', '<b>', '</b>', '</div>']
    """
    # Define the regular expression pattern to match HTML tags
    # Pattern explanation:
    # < - matches the opening angle bracket
    # [^>]+ - matches one or more characters that are not closing angle brackets
    # > - matches the closing angle bracket
    tag_pattern = r'<[^>]+>'
    
    # Use re.findall to find all occurrences of HTML tags in the content
    # This returns a list of all matching patterns
    tags = re.findall(tag_pattern, content)
    
    return tags
