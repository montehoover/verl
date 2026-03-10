import re
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_path(path: str) -> bool:
    """
    Validate if a given string is a valid HTTP or HTTPS URL path.
    
    Args:
        path (str): The input path to be validated.
        
    Returns:
        bool: True if the input is a valid HTTP/HTTPS path, False otherwise.
        
    Notes:
        The function uses a regular expression to check if the path:
        - Starts with 'http://' or 'https://'
        - Contains a valid domain name (no spaces or invalid characters)
        - May contain any valid URL path characters after the domain
    """
    # Regular expression pattern for validating HTTP/HTTPS URLs
    # ^https?:// - Matches 'http://' or 'https://' at the start
    # [^\s/$.?#] - Matches any character except whitespace and special chars for first domain char
    # [^\s]* - Matches any non-whitespace characters for the rest of the URL
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    # Simple validation with no nested conditions
    is_valid = bool(re.match(pattern, path))
    
    # Log the validation result
    if is_valid:
        logger.info(f"Path validation successful: '{path}'")
    else:
        logger.info(f"Path validation failed: '{path}'")
    
    return is_valid
