import re
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def verify_path_format(address: str) -> bool:
    """
    Verify if a given string is a valid HTTP or HTTPS URL.
    
    This function uses a regular expression pattern to validate whether
    the input string is a properly formatted HTTP or HTTPS URL.
    
    Args:
        address (str): The input string to validate as a URL.
        
    Returns:
        bool: True if the address is a valid HTTP/HTTPS URL, False otherwise.
        
    Examples:
        >>> verify_path_format("https://example.com")
        True
        >>> verify_path_format("http://subdomain.example.com/path")
        True
        >>> verify_path_format("ftp://example.com")
        False
        >>> verify_path_format("not a url")
        False
    """
    # Guard clause: log and return early if address is not a string
    if not isinstance(address, str):
        logger.warning(f"Invalid input type: expected str, got {type(address).__name__}")
        return False
    
    # Guard clause: log and return early if address is empty
    if not address:
        logger.debug("Empty address provided")
        return False
    
    # Regular expression pattern to match HTTP/HTTPS URLs
    # ^https?:// - Starts with http:// or https://
    # [^\s/$.?#] - First character after protocol must not be whitespace or special chars
    # . - Any character (ensures at least one character after protocol)
    # [^\s]* - Followed by any non-whitespace characters
    # $ - End of string
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    # Perform the validation
    is_valid = bool(re.match(pattern, address))
    
    # Log the validation result
    if is_valid:
        logger.info(f"Valid URL: {address}")
    else:
        logger.debug(f"Invalid URL: {address}")
    
    return is_valid
