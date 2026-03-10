"""URL path validation utilities.

This module provides a single public function, check_path_validity, which
validates whether a string is a well-formed HTTP or HTTPS URL by matching it
against a regular expression. It also logs each validation attempt and its
result for operational visibility.

Regex overview:
- Scheme: http:// or https://
- Host: one of:
  - 'localhost'
  - a domain name (labels of letters, digits, hyphens; ends with a TLD of
    2–63 letters)
  - an IPv4 address (four octets, 0–999; exact octet ranges are not enforced)
- Optional port: ':' followed by 2–5 digits (range validity not enforced)
- Optional path: zero or more slash-separated segments; excludes spaces, '?' or
  '#'
- Optional query: '?' followed by any non-space, non-# characters
- Optional fragment: '#' followed by any non-space characters
"""

import logging
import re

# Module-level logger. A NullHandler prevents "No handler found" warnings in
# library code. Applications can configure handlers/levels as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Compiled regex to validate HTTP/HTTPS URLs with optional port, path, query,
# and fragment. Uses VERBOSE for readability and inline documentation.
_HTTP_URL_PATTERN = re.compile(
    r"""
    ^
    (?:http|https)://                    # Scheme
    (?:                                  # Host
        localhost                        # - localhost
        |                                # - or domain name
        (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}
        |                                # - or IPv4 address
        (?:\d{1,3}\.){3}\d{1,3}
    )
    (?::\d{2,5})?                        # Optional port (2–5 digits)
    (?:/[^\s?#]*)*                       # Optional path segments
    (?:\?[^\s#]*)?                       # Optional query string
    (?:#[^\s]*)?                         # Optional fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def check_path_validity(path_string: str) -> bool:
    """Return True if the input is a valid HTTP/HTTPS URL, else False.

    The function performs a regex-based validation of the supplied string
    against a pattern that covers scheme, host (localhost, domain, IPv4),
    optional port, path, query, and fragment.

    It uses guard clauses for clarity and logs each validation attempt and its
    outcome. The function never raises; any unexpected error is caught and
    results in False.

    Args:
        path_string (str): The input string to validate as a path.

    Returns:
        bool: True if the path is formatted correctly, False otherwise.
    """
    if not isinstance(path_string, str):
        logger.info(
            "URL validation rejected (non-string input): %r",
            path_string,
        )
        return False

    logger.info("Validating URL: %s", path_string)

    try:
        is_valid = _HTTP_URL_PATTERN.fullmatch(path_string) is not None
    except Exception:
        # Ensure no exceptions escape from this function.
        logger.exception("Error while validating URL: %s", path_string)
        return False

    logger.info("Validation result for %s: %s", path_string, is_valid)
    return is_valid
