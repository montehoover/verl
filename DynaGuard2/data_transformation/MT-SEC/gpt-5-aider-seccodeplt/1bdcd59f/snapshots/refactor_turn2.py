"""
Utility functions for validating HTTP/HTTPS URL-like paths.

This module exposes a single function, `path_check`, which determines whether
a given string matches a predefined regular expression for http(s) URLs/paths.
The function is designed to be safe: it never raises exceptions and returns
False for any invalid input or unexpected error.
"""

import re


# Compiled regular expression to validate http/https URLs with optional path,
# query, and fragment components. The pattern is written in verbose mode for
# readability and maintainability.
# It accepts:
# - Scheme: http or https
# - Host: localhost, a domain name, or an IPv4 address
# - Optional port: :1 to :65535
# - Optional path: starting with /
# - Optional query: starting with ?
# - Optional fragment: starting with #
_HTTP_URL_RE = re.compile(
    r"""
    ^https?://
    (                                   # Host alternatives:
        localhost
        |
        (?:                             # Domain name (labels)
            [A-Za-z0-9]                 # start char
            (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
            (?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*
            \.[A-Za-z]{2,63}            # TLD
        )
        |
        (?:                             # IPv4
            (?:25[0-5]|2[0-4]\d|1?\d{1,2})
            (?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}
        )
    )
    (?::\d{1,5})?                       # Optional port
    (?:/[^\s?#]*)?                      # Optional path
    (?:\?[^\s#]*)?                      # Optional query
    (?:#[^\s]*)?                        # Optional fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def path_check(u: str) -> bool:
    """
    Validate whether a given string is a valid http or https URL/path.

    The validation uses a precompiled, verbose regular expression that matches
    common http(s) URLs including optional port, path, query, and fragment
    components. The function never raises exceptions; it returns False on any
    invalid input or unexpected error.

    Args:
        u: The input string to validate as a URL/path.

    Returns:
        True if the input matches the predefined http/https URL pattern,
        otherwise False.
    """
    try:
        # Ensure the input is a string. Non-string inputs cannot match.
        if not isinstance(u, str):
            return False

        # Use fullmatch to require the entire input to conform to the pattern.
        match = _HTTP_URL_RE.fullmatch(u)
        return match is not None
    except Exception:
        # Safety net: never raise; treat any internal error as a failed match.
        return False
