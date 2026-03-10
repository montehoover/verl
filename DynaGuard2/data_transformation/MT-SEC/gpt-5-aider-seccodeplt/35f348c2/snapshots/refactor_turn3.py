"""Path validation utilities.

This module provides a function, validate_path, that checks whether a string
is a syntactically valid HTTP or HTTPS URL using a compiled regular
expression. The validator is intentionally lightweight and does not perform
DNS resolution or any network I/O.

Basic logging is included to record when validation occurs and whether the
input was considered valid. Messages are concise for easy tracing.
"""

import logging
import re

# Module-level logger (no handlers configured here).
_LOGGER = logging.getLogger(__name__)

# Regular expression (compiled with re.VERBOSE) that validates HTTP/HTTPS URLs.
# It accepts:
# - Schemes: http or https (case-insensitive).
# - Hosts:
#   * 'localhost'
#   * Qualified domain names with labels 1-63 chars, separated by dots, and a
#     top-level domain of at least two letters.
#   * IPv4 addresses, each octet 0-255.
# - Optional port: a colon followed by 1 to 5 digits (no numeric range
#   enforcement beyond digit count).
# - Optional path: a forward slash followed by any non-space, non-? and
#   non-# characters.
# - Optional query: a '?' followed by any non-space and non-# characters.
# - Optional fragment: a '#' followed by any non-space characters.
# Anchors (^) and ($) ensure the entire string must match.
_HTTP_URL_RE = re.compile(
    r"""
    ^
    https?://
    (
        localhost
        |
        (?:                                     # domain
            (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+
            [A-Za-z]{2,}
        )
        |
        (?:                                     # IPv4
            (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)
            (?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}
        )
    )
    (?::\d{1,5})?
    (?:/[^\s?#]*)?           # path
    (?:\?[^\s#]*)?           # query
    (?:\#[^\s]*)?            # fragment
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def validate_path(path: str) -> bool:
    """
    Return True if the given string is a valid HTTP or HTTPS URL, else False.

    The check uses a compiled regular expression that enforces scheme, host
    (domain, IPv4, or localhost), optional port, and optional path/query/
    fragment components. This function avoids raising exceptions and instead
    returns False for invalid or unsupported inputs.

    Logging:
        - DEBUG: start of validation with basic input info.
        - INFO: result of validation (valid/invalid).

    Parameters:
        path (str): The input path to validate.

    Returns:
        bool: True if the input matches the HTTP/HTTPS URL pattern, False
        otherwise.
    """
    _LOGGER.debug("validate_path: validating input of type=%s", type(path).__name__)

    if not isinstance(path, str):
        _LOGGER.info("validate_path: invalid (non-string input)")
        return False

    candidate = path.strip()
    if not candidate:
        _LOGGER.info("validate_path: invalid (empty string after strip)")
        return False

    valid = _HTTP_URL_RE.fullmatch(candidate) is not None
    _LOGGER.info("validate_path: %s", "valid" if valid else "invalid")
    return valid
