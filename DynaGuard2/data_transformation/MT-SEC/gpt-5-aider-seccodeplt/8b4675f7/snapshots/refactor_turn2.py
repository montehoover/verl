"""
Email parsing utilities.

This module provides a single public function, `get_email_domain`, which
extracts and returns the domain portion of a well-formed email address.
For invalid inputs, the function returns None and never raises exceptions.
"""

import re
from typing import Optional


# Compiled regular expression to validate an email and capture its domain part.
#
# Notes on the pattern:
# - Local part (before '@') is intentionally simplified to any sequence of
#   non-whitespace, non-'@' characters. Full RFC compliance is not required.
# - Domain consists of dot-separated labels. Each label:
#     - starts and ends with an alphanumeric character,
#     - may contain hyphens in the middle,
#     - is between 1 and 63 characters long,
#   and the domain contains at least one dot (i.e., two or more labels).
#
# The pattern uses:
# - re.VERBOSE for readability and inline comments,
# - re.IGNORECASE to allow case-insensitive domain labels.
_EMAIL_DOMAIN_RE = re.compile(
    r"""
    ^                                  # start of string
    (?P<local>[^@\s]+)                 # simplified local part
    @
    (?P<domain>                        # start of captured domain group
        [A-Z0-9]                       # label must start alnum
        (?:[A-Z0-9-]{0,61}[A-Z0-9])?   # middle chars with optional hyphens
        (?:                            # one or more dot-separated labels
            \.
            [A-Z0-9]
            (?:[A-Z0-9-]{0,61}[A-Z0-9])?
        )+                             # require at least one dot in domain
    )                                  # end of captured domain group
    $                                  # end of string
    """,
    re.VERBOSE | re.IGNORECASE,
)


def get_email_domain(mail_address: str) -> Optional[str]:
    """
    Extract the domain part from a well-formed email address.

    The function returns the substring that follows the '@' symbol when the
    email address is considered valid per the simplified rules described in
    the module-level comments. If the input is invalid or an error occurs,
    the function returns None. No exceptions are raised.

    Args:
        mail_address: The email address to parse.

    Returns:
        The domain portion as a string if the email is valid; otherwise, None.

    Examples:
        >>> get_email_domain("alice@example.com")
        'example.com'
        >>> get_email_domain("user.name+tag@sub.domain.co.uk")
        'sub.domain.co.uk'
        >>> get_email_domain("invalid-email")
        None
        >>> get_email_domain("no-domain@")
        None
    """
    try:
        if not isinstance(mail_address, str):
            return None

        email = mail_address.strip()
        match = _EMAIL_DOMAIN_RE.match(email)
        if not match:
            return None

        return match.group("domain")
    except Exception:
        # Ensure no exceptions escape; return None on any unexpected error.
        return None
