"""
Email parsing utilities.

This module provides a single public function, `get_email_domain`, which
extracts and returns the domain portion of a well-formed email address.
For invalid inputs, the function returns None and never raises exceptions.

Logging:
    This module uses the standard logging library. It logs each processed
    email address and whether domain extraction succeeded. By default, a
    NullHandler is attached so importing this module will not configure
    logging. Applications should configure logging as needed.
"""

import logging
import re
from typing import Optional

__all__ = ["get_email_domain"]

# Module logger. Applications can configure handlers/levels.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

    Logging:
        - DEBUG: When parsing starts and the outcome details.
        - INFO:  Success or invalid email notifications.
        - ERROR: Unexpected exceptions encountered.

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
            logger.info(
                "Invalid email type encountered: %r", type(mail_address).__name__
            )
            return None

        email = mail_address.strip()
        logger.debug("Parsing email: %r", email)

        match = _EMAIL_DOMAIN_RE.match(email)
        domain = match.group("domain") if match else None

        if domain is None:
            logger.info("Invalid email address: %r", email)
            return None

        logger.info("Extracted domain '%s' from email %r", domain, email)
        return domain

    except Exception:
        # Ensure no exceptions escape; return None on any unexpected error.
        logger.exception("Unexpected error while parsing email.")
        return None
