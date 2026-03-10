import re
import logging

logger = logging.getLogger(__name__)


def fetch_email_domain(address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.

    Args:
        address: str, the email address to process and analyze.

    Returns:
        If the email address is valid, return the domain as a string.
        If it is invalid, return None.
        Exceptions should not be raised by the function.
    """
    # This regular expression is used to validate the structure of an email
    # address and to capture the domain name.
    # It checks for:
    # 1. Local part: One or more characters consisting of alphanumeric
    #    characters, dots (.), underscores (_), percent signs (%),
    #    plus signs (+), or hyphens (-).
    # 2. At symbol: A literal '@' character.
    # 3. Domain part (captured):
    #    - Domain name: One or more characters consisting of alphanumeric
    #      characters, dots (.), or hyphens (-).
    #    - Top-Level Domain (TLD): A literal dot (.) followed by at least
    #      two alphabetic characters (e.g., .com, .org).
    # The entire string must match this pattern (due to ^ and $ anchors).
    # The domain part (e.g., "example.com") is captured in a group.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"

    logger.info(f"Attempting to extract domain from email: '{address}'")
    match = re.fullmatch(email_regex, address)

    if not match:
        logger.warning(f"Invalid email format or no domain found for: '{address}'")
        # If the email address does not match the pattern, it's considered
        # invalid for the purpose of domain extraction.
        return None

    # If we reach here, the email address matches the pattern.
    # group(1) of the match object contains the captured domain part.
    domain = match.group(1)
    logger.info(f"Successfully extracted domain '{domain}' from email: '{address}'")
    return domain
