import re
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if no handlers exist
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def append_user_header(user_header: str) -> str:
    # Log the input header
    logger.debug(f"Input header received: {repr(user_header)}")
    
    # Guard clause: Check if user_header is None or empty
    if not user_header:
        logger.warning("Empty or None header received")
        return "X-Custom-Header: "
    
    # Guard clause: Check if user_header is not a string
    if not isinstance(user_header, str):
        logger.info(f"Non-string header received, converting to string: {type(user_header)}")
        user_header = str(user_header)
    
    # Guard clause: Remove invalid characters if present
    if re.search(r'[\r\n\x00-\x1f\x7f]', user_header):
        logger.warning(f"Invalid characters found in header, sanitizing: {repr(user_header)}")
        user_header = re.sub(r'[\r\n\x00-\x1f\x7f]', '', user_header)
        logger.debug(f"Sanitized header: {repr(user_header)}")
    
    # Create the full header string with the custom header
    full_header = f"X-Custom-Header: {user_header}"
    
    # Log the final header string
    logger.debug(f"Final header string: {repr(full_header)}")
    
    return full_header
