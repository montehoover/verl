import re
import logging

# Initialize logger
logger = logging.getLogger(__name__)
# Basic configuration for the logger (can be configured externally as well)
# This will output logs to the console.
# For a real application, you might want to configure this in a more sophisticated way.
if not logger.handlers: # Avoid adding multiple handlers if the module is reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Set default logging level

def append_user_header(user_header: str) -> str:
    """
    Inserts a custom header into the output headers of a response.

    Args:
        user_header: str, representing the custom header value input by the user.

    Returns:
        str, which contains the full header string including the custom header value.

    Raises:
        TypeError: If user_header is not a string.
        ValueError: If user_header is empty or contains invalid characters (newline, carriage return).
    """
    logger.info(f"Received user_header: '{user_header}'")

    if not isinstance(user_header, str):
        logger.error("TypeError: Header value must be a string.")
        raise TypeError("Header value must be a string.")

    if not user_header:
        logger.error("ValueError: Header value cannot be empty.")
        raise ValueError("Header value cannot be empty.")

    if re.search(r"[\n\r]", user_header):
        logger.error("ValueError: Header value cannot contain newline or carriage return characters.")
        raise ValueError("Header value cannot contain newline or carriage return characters.")

    header_name = "X-User-Header"
    final_header = f"{header_name}: {user_header}"
    logger.info(f"Generated final header: '{final_header}'")
    return final_header
