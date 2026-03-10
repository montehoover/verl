"""
Utilities for validating HTTP/HTTPS paths using a compiled regular expression.

Public API:
- check_path(input_path: str) -> bool
"""

import re


# Compiled regex for validating HTTP/HTTPS URLs.
# The pattern supports:
# - Scheme: http or https
# - Host: domain (e.g., example.com), localhost, or IPv4 address
# - Optional port: :80, :443, etc.
# - Optional path/query/fragment: begins with '/', followed by non-whitespace
_HTTP_PATH_RE = re.compile(
    r"""
    ^                             # start of string
    (?:http|https)://             # scheme
    (?:
        localhost                 # localhost
        |
        (?:                       # domain name with TLD (letters only)
            (?:                   # one or more labels ending with a dot
                [A-Za-z0-9]
                (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
                \.
            )+
            [A-Za-z]{2,63}        # TLD: 2-63 letters
        )
        |
        (?:\d{1,3}\.){3}\d{1,3}   # IPv4 address
    )
    (?::\d{2,5})?                 # optional port
    (?:/[^ \t\r\n]*)?             # optional path (no whitespace)
    $                             # end of string
    """,
    re.VERBOSE,
)


def _matches_http_https_pattern(value: str) -> bool:
    """
    Return True if the given string matches the HTTP/HTTPS URL pattern.

    This helper isolates the regex logic so it is reusable and independently
    testable.
    """
    return _HTTP_PATH_RE.match(value) is not None


def check_path(input_path: str) -> bool:
    """
    Determine whether the given string is a valid HTTP or HTTPS path.

    Args:
        input_path: The string to evaluate.

    Returns:
        True if the input matches the HTTP/HTTPS path structure, False otherwise.

    Notes:
        This function is defensive and will never raise; any unexpected error
        results in a False return value.
    """
    try:
        if not isinstance(input_path, str):
            return False

        return _matches_http_https_pattern(input_path)
    except Exception:
        # Ensure no exceptions escape this function.
        return False
