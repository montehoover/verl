"""
Utilities for validating HTTP/HTTPS URLs.

This module provides:
- _URL_RE: a compiled regular expression describing a pragmatic HTTP/HTTPS URL.
- is_valid_path: a function that checks whether a string matches that pattern.

The pattern covers the following:
- http and https schemes
- hostnames (including subdomains), IPv4 addresses, and localhost
- optional port number
- optional path segments
- optional query string
- optional fragment

Note: This is not a full RFC 3986 validator but is suitable for common cases.
"""

import re

# Compiled regular expression to validate HTTP/HTTPS URLs. Uses re.VERBOSE to
# allow whitespace and inline comments for readability.
_URL_RE = re.compile(
    r"""
    ^
    https?://
    (?:
        # IPv4 address (0.0.0.0 to 255.255.255.255)
        (?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)
        |
        # Hostname with optional subdomains.
        # - Labels start and end with alphanumeric characters.
        # - Interior characters may include hyphens.
        # - TLD is at least two letters.
        (?:
            (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)
            \.
        )+
        (?:[A-Za-z]{2,})
        |
        # localhost
        localhost
    )
    # Optional port (1 to 5 digits).
    (?::\d{1,5})?
    # Optional path segments with common unreserved and reserved URL characters.
    (?:/[A-Za-z0-9._~!$&'()*+,;=:@%\-]*)*
    # Optional query string.
    (?:\?[A-Za-z0-9._~!$&'()*+,;=:@%/\-]*)?
    # Optional fragment.
    (?:\#[A-Za-z0-9._~!$&'()*+,;=:@%/\-]*)?
    $
    """,
    re.VERBOSE,
)


def is_valid_path(site_path: str) -> bool:
    """
    Return True if the given string is a valid HTTP/HTTPS URL, otherwise False.

    The check is performed by matching the input against a compiled regular
    expression that accepts common URL forms, including optional ports, paths,
    queries, and fragments. This function is defensive and will not raise
    exceptions; any unexpected error results in False.

    Args:
        site_path: The string to evaluate as a potential valid HTTP/HTTPS URL.

    Returns:
        True if the input matches the URL pattern; otherwise, False.
    """
    try:
        if not isinstance(site_path, str):
            return False

        # Normalize by trimming surrounding whitespace.
        candidate = site_path.strip()
        if not candidate:
            return False

        return _URL_RE.match(candidate) is not None
    except Exception:
        # Guarantee that no exceptions escape; return False on any error.
        return False
