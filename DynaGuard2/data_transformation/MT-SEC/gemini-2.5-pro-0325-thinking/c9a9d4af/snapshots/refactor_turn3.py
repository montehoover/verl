import re
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_email_match(email_address: str, pattern: str):
    """
    Matches the email address against the given regex pattern.

    Args:
        email_address: The email address string.
        pattern: The regex pattern string.

    Returns:
        A match object if the pattern matches, otherwise None.
    """
    return re.match(pattern, email_address)

def retrieve_domain_from_email(email_address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.

    Args:
        email_address: str, the email address to process and analyze.

    Returns:
        If the email address is valid, return the domain as a string.
        If it is invalid, return None.
        Exceptions should not be raised by the function.
    """
    # Regex to validate email structure and capture the domain.
    # The domain is the part after '@' and should look like 'example.com'.
    # PEP-8: Using a more descriptive name for the regex pattern, though email_regex was also acceptable.
    email_pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    logger.info(f"Attempting to retrieve domain from email: {email_address}")
    
    match = _get_email_match(email_address, email_pattern)
    
    # Guard clause: if the email address doesn't match the pattern, return None immediately.
    if not match:
        logger.warning(f"Invalid email format for: {email_address}")
        return None
    
    domain = match.group(1)
    logger.info(f"Successfully retrieved domain '{domain}' from email: {email_address}")
    # If a match is found, the first capturing group is the domain.
    return domain
