import re
import logging


# Configure logging
logger = logging.getLogger(__name__)


def get_html_tag_pattern() -> str:
    """
    Return the regular expression pattern for matching HTML tags.
    
    Returns:
        str: Regular expression pattern that matches HTML tags including
             opening tags, closing tags, and self-closing tags.
    
    Pattern explanation:
        < : Matches the opening angle bracket
        [^>]+ : Matches one or more characters that are not closing angle brackets
        > : Matches the closing angle bracket
    """
    return r'<[^>]+>'


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
    # Log the input content
    logger.debug(f"Processing HTML content of length: {len(html_content)}")
    logger.debug(f"Input content: {html_content[:100]}{'...' if len(html_content) > 100 else ''}")
    
    # Get the HTML tag pattern
    pattern = get_html_tag_pattern()
    
    # Find all matches of HTML tags in the html_content string
    tags = re.findall(pattern, html_content)
    
    # Log the results
    logger.info(f"Found {len(tags)} HTML tags")
    logger.debug(f"Tags found: {tags}")
    
    return tags
