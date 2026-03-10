import re
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def include_custom_header(custom_value: str) -> str:
    """
    Inserts a custom header into the response headers, where the header value is supplied by the user.

    The user input must be a valid string suitable for an HTTP header field.

    Args:
        custom_value: str, serving as the user-defined value for the custom header.

    Returns:
        str, representing the full header string which incorporates the sanitized custom header value.

    Raises:
        re.error: While no explicit exceptions are raised by this function's logic, 
                  improper regex patterns (not applicable here with fixed pattern) 
                  or internal 're' module issues could cause 're.error' exceptions.
    """
    logger.info(f"Received custom_value: '{custom_value}'")

    # Sanitize the custom value to remove characters that could break HTTP header structure (e.g., CR, LF)
    # A more comprehensive sanitization might involve allowing only specific characters (e.g., printable ASCII).
    sanitized_value = re.sub(r'[\r\n]', '', custom_value)
    if custom_value != sanitized_value:
        logger.info(f"Sanitized custom_value to: '{sanitized_value}'")
    else:
        logger.info("custom_value did not require sanitization.")

    # Define the custom header name. This can be changed as needed.
    header_name = "X-Custom-Header"

    header_string = f"{header_name}: {sanitized_value}"
    logger.info(f"Constructed header string: '{header_string}'")
    
    return header_string
