import re
import logging


def get_html_tag_pattern() -> str:
    """
    Return the regex pattern for matching HTML tags.
    
    Returns:
        str: Regular expression pattern for HTML tags.
    """
    # Pattern explanation:
    # < - matches the opening angle bracket
    # [^>]+ - matches one or more characters that are not closing angle brackets
    # > - matches the closing angle bracket
    return r'<[^>]+>'


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
    # Log the input HTML string
    logging.debug(f"Parsing HTML string: {html}")
    
    # Get the regex pattern
    pattern = get_html_tag_pattern()
    
    # Find all matches
    tags = re.findall(pattern, html)
    
    # Log the extracted tags
    logging.debug(f"Extracted {len(tags)} tags: {tags}")
    
    return tags
