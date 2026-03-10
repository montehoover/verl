import re
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def get_html_tag_pattern() -> str:
    """
    Returns the regular expression pattern for matching HTML tags.
    
    This pattern matches opening tags (e.g., <div>), closing tags (e.g., </div>),
    and self-closing tags (e.g., <br />).
    
    Returns:
        str: Regular expression pattern for HTML tag matching.
        
    Pattern explanation:
        < - matches the opening angle bracket
        [^>]+ - matches one or more characters that are not closing angle brackets
        > - matches the closing angle bracket
    """
    return r'<[^>]+>'


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
    # Log the input content for debugging purposes
    logger.debug(f"Processing HTML content of length: {len(content)}")
    logger.debug(f"Content preview: {content[:100]}..." if len(content) > 100 else f"Content: {content}")
    
    # Get the HTML tag pattern from the dedicated function
    tag_pattern = get_html_tag_pattern()
    
    # Use re.findall to find all occurrences of HTML tags in the content
    # This returns a list of all matching patterns
    tags = re.findall(tag_pattern, content)
    
    # Log the results for debugging
    logger.info(f"Identified {len(tags)} HTML tags")
    logger.debug(f"Tags found: {tags}")
    
    return tags
