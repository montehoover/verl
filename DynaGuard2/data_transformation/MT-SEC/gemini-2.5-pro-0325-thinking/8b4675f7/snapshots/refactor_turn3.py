import re
import logging
from typing import Optional


# Initialize logger for this module
logger = logging.getLogger(__name__)

# Regular expression to validate email and capture the domain.
# This pattern aims to match common email formats but may not cover all
# valid or esoteric email address specifications (e.g., internationalized domain names
# without Punycode, or quoted local parts).
#
# Pattern breakdown:
# ^                     - Asserts the start of the string.
# [a-zA-Z0-9._%+-]+     - Matches the local part of the email (before '@').
#                         Allows one or more alphanumeric characters, dots,
#                         underscores, percent signs, plus signs, or hyphens.
# @                     - Matches the literal "@" symbol.
# (                     - Starts a capturing group for the domain part.
#   [a-zA-Z0-9.-]+      - Matches the domain name. Allows one or more
#                         alphanumeric characters, dots, or hyphens.
#   \.                  - Matches the literal dot before the top-level domain.
#   [a-zA-Z]{2,}       - Matches the top-level domain (TLD), e.g., .com, .org.
#                         Requires at least two alphabetic characters.
# )                     - Ends the capturing group for the domain part.
# $                     - Asserts the end of the string.
_EMAIL_DOMAIN_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$")


def get_email_domain(mail_address: str) -> Optional[str]:
    """
    Extracts the domain portion from an email address using regular expressions.
    Logs the input, processing steps, and the outcome.

    Args:
        mail_address: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid and a domain is found,
        returns the domain portion as a string. Otherwise, returns None.
        The function should not raise any exceptions.
    """
    logger.debug(f"Processing email address: '{mail_address}'")

    if not isinstance(mail_address, str):
        logger.warning(
            f"Invalid input type for mail_address: {type(mail_address).__name__}. Expected str. Email: '{mail_address}'"
        )
        return None
    
    # Attempt to match the entire mail_address string against the pre-compiled regex.
    match = _EMAIL_DOMAIN_REGEX.fullmatch(mail_address)
    
    if match:
        domain = match.group(1)
        logger.info(f"Successfully extracted domain '{domain}' from '{mail_address}'.")
        return domain
    else:
        logger.info(f"No valid domain found in '{mail_address}'. Email does not match expected pattern.")
        return None
