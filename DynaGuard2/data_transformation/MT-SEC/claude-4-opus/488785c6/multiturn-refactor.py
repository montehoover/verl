import re
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def add_safe_header(custom_value: str) -> str:
    """
    Add a safe custom header to HTTP response headers.
    
    Sanitizes user-supplied values to prevent header injection attacks
    and ensure compliance with valid HTTP header rules.
    
    Args:
        custom_value (str): The user-provided custom header value.
        
    Returns:
        str: The final header string including the sanitized custom header value.
        
    Note:
        Does not raise any specific exceptions, but improper input could
        indirectly raise 're.error'.
    """
    # Log the original input value
    logger.debug(f"Processing custom header value: {repr(custom_value)}")
    
    # Regex pattern explanation:
    # [^...] - Matches any character NOT in the set
    # a-zA-Z0-9 - Allows alphanumeric characters (letters and numbers)
    # \s - Allows whitespace characters
    # \- - Allows hyphens (escaped because - has special meaning in character sets)
    # _ - Allows underscores
    # This pattern removes any characters that could be used for header injection,
    # including newlines (\n), carriage returns (\r), and other control characters
    sanitized_value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', custom_value)
    
    # Log if any characters were removed during sanitization
    if sanitized_value != custom_value:
        logger.warning(
            f"Removed invalid characters from header value. "
            f"Original: {repr(custom_value)}, Sanitized: {repr(sanitized_value)}"
        )
    
    # Remove leading and trailing whitespace to ensure clean header value
    sanitized_value = sanitized_value.strip()
    
    # Limit header value length to prevent excessively long headers
    # that could cause issues with HTTP servers or clients
    MAX_HEADER_LENGTH = 100
    if len(sanitized_value) > MAX_HEADER_LENGTH:
        logger.warning(
            f"Header value truncated from {len(sanitized_value)} to "
            f"{MAX_HEADER_LENGTH} characters"
        )
    sanitized_value = sanitized_value[:MAX_HEADER_LENGTH]
    
    # Construct the final header string with a custom header name
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    # Log the final header string
    logger.info(f"Generated header: {header_string}")
    
    return header_string
