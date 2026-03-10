import re
import logging

# Configure basic logging if no handlers are configured
# This is useful for scripts/libraries so users see logs by default
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def validate_path(path: str) -> bool:
    """
    Checks if a given string is a valid http or https path using a regular expression.

    Args:
        path: str, the input path to be validated.

    Returns:
        True if the input is a valid path, False otherwise.
    """
    # Regular expression for validating HTTP/HTTPS URLs
    # This regex checks for:
    # - http:// or https://
    # - Optional www.
    # - Domain name (alphanumeric, hyphen, dot)
    # - Optional port number (e.g., :8080)
    # - Optional path (alphanumeric, hyphen, underscore, slash, dot, question mark, equals, ampersand, hash)
    # - Optional query parameters
    # - Optional fragment identifier
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    logger.info(f"Attempting to validate path: '{path}'")
    # Return True if the path matches the pattern, False otherwise.
    is_valid = bool(re.match(pattern, path))
    
    if is_valid:
        logger.info(f"Validation successful for path: '{path}'")
    else:
        logger.warning(f"Validation failed for path: '{path}'")
        
    return is_valid
