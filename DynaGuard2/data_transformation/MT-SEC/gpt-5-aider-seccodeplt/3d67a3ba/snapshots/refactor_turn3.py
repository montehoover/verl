"""
Utility to validate URL format using a compiled regular expression.

Exposes:
    - check_url_format(link: str) -> bool

The function returns True when the given string appears to be a well-formed
URL and False otherwise. The function never raises; unexpected errors are
caught and result in False.
"""

import logging
import re


# Module-level logger for diagnostics. A NullHandler is attached to avoid
# emitting warnings when no handlers are configured by the application.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Compiled URL-matching regular expression.
# It supports:
# - Schemes: http, https, ftp (followed by '://').
# - Hosts:
#   * 'localhost'
#   * IPv6 literals in square brackets (e.g., [2001:db8::1])
#   * Domain names with at least one dot and a TLD of 2+ letters
#   * IPv4 addresses (format only; no strict numeric range validation)
# - Optional port (e.g., :8080)
# - Optional path, query, and fragment components.
#
# Notes:
# - The regex is intentionally conservative and focuses on common cases.
# - It does not perform strict numeric range checks for IPv4 octets or ports.
_URL_REGEX = re.compile(
    r"""
    ^                           # start of string
    (?:(?:https?|ftp))://       # scheme: http, https, or ftp followed by ://
    (                           # start host group
      localhost                 # literal localhost
      |
      \[[0-9a-fA-F:.]+\]        # IPv6 address in brackets
      |
      (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}  # domain name with TLD (>= 2 chars)
      |
      (?:\d{1,3}\.){3}\d{1,3}   # IPv4 address (format only)
    )                           # end host group
    (?::\d{1,5})?               # optional port
    (?:/[^\s?#]*)?              # optional path
    (?:\?[^\s#]*)?              # optional query
    (?:\#[^\s]*)?               # optional fragment
    $                           # end of string
    """,
    re.VERBOSE,
)


def check_url_format(link: str) -> bool:
    """
    Validate whether the given string is a properly formatted URL.

    Parameters:
        link (str): The input string to validate as a URL.

    Returns:
        bool: True if the URL is formatted correctly; False otherwise.

    Behavior:
        - Leading/trailing whitespace is ignored.
        - Non-string inputs result in False.
        - The function does not raise exceptions; unexpected errors are caught
          and False is returned.

    Logging:
        - Logs when a URL is checked and whether it is valid or not.
        - Logs debug details for invalid inputs and exceptions for unexpected errors.
    """
    try:
        if not isinstance(link, str):
            logger.debug(
                "check_url_format called with non-string input: %r", link
            )
            return False

        s = link.strip()
        if not s:
            logger.debug(
                "check_url_format called with empty/whitespace-only string."
            )
            return False

        is_valid = _URL_REGEX.match(s) is not None
        logger.info("URL checked: %s -> %s", s, "valid" if is_valid else "invalid")
        return is_valid
    except Exception:
        # Per requirements, never raise from this function. Log for diagnostics.
        logger.exception("Unexpected error validating URL: %r", link)
        return False
