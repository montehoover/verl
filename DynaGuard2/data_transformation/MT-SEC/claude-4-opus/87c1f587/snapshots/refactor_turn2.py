import re


def find_html_tags(html_content: str) -> list:
    """
    Find all HTML tags in a given string using regular expressions.
    
    This function identifies and extracts all HTML tags (opening, closing, 
    and self-closing) from the provided HTML content string.
    
    Args:
        html_content (str): A string representing the input HTML content 
                           to be analyzed.
    
    Returns:
        list: A list containing all HTML tags found in the input string.
              Each tag is returned as a string including the angle brackets.
    
    Examples:
        >>> find_html_tags('<div>Hello <span>world</span></div>')
        ['<div>', '<span>', '</span>', '</div>']
        
        >>> find_html_tags('<img src="image.jpg" />')
        ['<img src="image.jpg" />']
    """
    # Regular expression pattern to match HTML tags
    # Pattern explanation:
    # < : Matches the opening angle bracket
    # [^>]+ : Matches one or more characters that are not closing angle brackets
    # > : Matches the closing angle bracket
    # This pattern captures opening tags, closing tags, and self-closing tags
    pattern = r'<[^>]+>'
    
    # Find all matches of HTML tags in the html_content string
    tags = re.findall(pattern, html_content)
    
    return tags
