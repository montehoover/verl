import re
import logging

# Configure basic logging if no handlers are configured
# This is a good practice for libraries, but for applications,
# logging configuration is often done at the application entry point.
# For this exercise, we'll add a basic configuration.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def verify_path_format(address: str) -> bool:
    """
    Verifies if a given string is a valid http or https path based on a regular expression pattern.

    Args:
        address: str, the input string to validate as a path.

    Returns:
        True if the path is formatted correctly, and False otherwise.
        Raises no exceptions.
    """
    # Regex to check for valid http or https URL.
    # This regex is a common pattern and covers most cases but might not be
    # exhaustive for all edge cases of URLs.
    pattern = re.compile(
        r'^(https?://)'          # Scheme: http:// or https://
        r'([A-Za-z0-9.-]+)'      # Authority: domain name or IP address.
                                 # Allows for alphanumeric characters, dots, and hyphens.
        r'(:\d+)?'               # Port: optional, starts with a colon, followed by digits.
        r'(/[^?#]*)?'            # Path: optional, starts with a slash, can contain any characters
                                 # except '?' or '#'.
        r'(\?[^#]*)?'            # Query: optional, starts with '?', can contain any characters
                                 # except '#'.
        r'(#.*)?$',              # Fragment: optional, starts with '#', can contain any characters.
                                 # '$' asserts position at the end of the string.
        re.IGNORECASE            # Makes the regex case-insensitive.
    )

    # Attempt to match the entire string against the compiled pattern.
    # re.fullmatch() returns a match object if the whole string matches, otherwise None.
    logger.info(f"Validating path: {address}")
    
    is_valid = bool(re.fullmatch(pattern, address))
    
    if is_valid:
        logger.info(f"Path '{address}' is valid.")
    else:
        logger.warning(f"Path '{address}' is invalid.")
        
    return is_valid
