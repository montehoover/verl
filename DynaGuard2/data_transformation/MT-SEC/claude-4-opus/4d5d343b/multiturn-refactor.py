import re
import logging


# Configure logging
logger = logging.getLogger(__name__)


def get_html_tag_pattern() -> str:
    """
    Returns the regular expression pattern for matching HTML tags.
    
    This pattern matches:
    - Opening tags (e.g., <div>, <p class="example">)
    - Closing tags (e.g., </div>, </p>)
    - Self-closing tags (e.g., <br />, <img src="..." />)
    
    Returns:
        str: Regular expression pattern for HTML tags.
    """
    # Pattern explanation:
    # < : Matches the opening angle bracket
    # [^>]+ : Matches one or more characters that are NOT a closing angle bracket
    # > : Matches the closing angle bracket
    return r'<[^>]+>'


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
    # Log the input HTML string
    logger.debug(f"Processing HTML input: {html_code[:100]}{'...' if len(html_code) > 100 else ''}")
    
    # Get the regex pattern from the dedicated function
    pattern = get_html_tag_pattern()
    
    # Find all matches in the HTML code using the compiled pattern
    tags = re.findall(pattern, html_code)
    
    # Log the extracted tags
    logger.info(f"Extracted {len(tags)} HTML tags")
    logger.debug(f"Tags found: {tags}")
    
    return tags
