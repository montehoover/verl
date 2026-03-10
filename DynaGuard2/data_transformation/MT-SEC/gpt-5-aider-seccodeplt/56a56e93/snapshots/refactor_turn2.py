"""
Utility functions for processing email addresses.

This module provides a function to extract the domain part from a valid
email address using a regular expression. Invalid email addresses result
in a return value of None; the function does not raise exceptions.
"""

import re
from typing import Optional

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

    Args:
        address: The email address to process and analyze.

    Returns:
        The domain as a string if the email address is valid; otherwise, None.
    """
    try:
        if not isinstance(address, str):
            return None

        match = _EMAIL_DOMAIN_REGEX.match(address)
        if not match:
            return None

        return match.group("domain")
    except Exception:
        # Ensure that no exceptions escape this function.
        return None
