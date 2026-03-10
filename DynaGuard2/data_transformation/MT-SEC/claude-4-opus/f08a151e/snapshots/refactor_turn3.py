import re
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_html_tags(html_input: str) -> list:
    """Extract HTML tags from a provided string using regular expressions.
    
    Args:
        html_input: An HTML-formatted string as input.
        
    Returns:
        A list containing the tags that are present within the input HTML string.
    """
    # Log the input HTML string
    logger.info(f"Processing HTML input: {html_input}")
    
    # Define the regular expression pattern for matching HTML tags
    # This pattern captures tags like <tag>, </tag>, and self-closing tags like <tag/>
    html_tag_pattern = r'<[^>]+>'
    
    # Extract all HTML tags from the input string
    extracted_tags = re.findall(html_tag_pattern, html_input)
    
    # Log the extracted tags
    logger.info(f"Extracted tags: {extracted_tags}")
    
    return extracted_tags
