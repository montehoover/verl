"""
Utility functions for processing email addresses.

This module provides a function to extract the domain part from a valid
email address using a regular expression. Invalid email addresses result
in a return value of None; the function does not raise exceptions.

Logging:
- Each processed email address is logged.
- Successful extractions log the extracted domain and the source email.
- Invalid inputs or unexpected errors are also logged.
"""

import logging
import re
from typing import Optional

# Module-level logger. Library code does not configure logging; the
# application using this module should configure handlers/levels.
logger = logging.getLogger(__name__)

# Regular expression (compiled with re.VERBOSE for readability) to validate
# an email address and capture its domain part.
#
# The pattern ensures:
# - A "local part" composed of a reasonable set of allowed characters.
# - A single '@' separator.
# - A domain composed of one or more labels separated by dots:
#   - Each label starts and ends with an alphanumeric character.
#   - Labels may include hyphens in the middle (but not at the start/end).
# - A top-level domain (TLD) with at least two letters.
_EMAIL_DOMAIN_REGEX = re.compile(
    r"""
    ^                                      # start of string
    [A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+       # local part (allowed characters)
    @                                      # at-sign separator
    (?P<domain>                            # begin named group 'domain'
      (?:                                  # one or more domain labels followed by a dot
        [A-Za-z0-9]                        # label starts with alphanumeric
        (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])? # middle may contain hyphens
        \.                                 # dot separator
      )+
      [A-Za-z]{2,}                         # TLD (at least two letters)
    )
    $                                      # end of string
    """,
    re.VERBOSE,
)


def fetch_email_domain(address: str) -> Optional[str]:
    """
    Extract the domain from a valid email address.

    This function validates the structure of the provided email address
    using a regular expression. If the email is valid, it returns the
    domain portion (text following the '@' sign). For invalid inputs or
    on any error, it returns None. The function does not raise exceptions.

    Logging is emitted for each processed address, including whether a domain
    was extracted or the input was invalid.

    Args:
        address: The email address to process and analyze.

    Returns:
        The domain as a string if the email address is valid; otherwise, None.
    """
    try:
        logger.info("Processing email address: %s", address)

        if not isinstance(address, str):
            logger.info("Invalid type for email address: %r", address)
            return None

        if match := _EMAIL_DOMAIN_REGEX.match(address):
            domain = match.group("domain")
            logger.info(
                "Extracted domain '%s' from email address '%s'", domain, address
            )
            return domain

        logger.info("Invalid email address structure: %s", address)
        return None
    except Exception:
        # Ensure that no exceptions escape this function; include traceback.
        logger.exception(
            "Unexpected error while processing email address: %r", address
        )
        return None
