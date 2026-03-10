"""
Utilities for verifying whether a string is a valid HTTP or HTTPS URL
based on a regular expression.

Public API:
    - verify_path_format(address: str) -> bool

The verifier is defensive: it returns True or False and never raises
exceptions.
"""

import re


# Compiled regular expression for HTTP/HTTPS URLs.
#
# Overall structure:
#   ^https?://                                # http:// or https://
#   (?:                                       # Start host alternatives
#       (?:                                   # Domain name:
#           [a-z0-9]                          #   label starts with alnum
#           (?:[a-z0-9-]{0,61}[a-z0-9])?      #   label middle and end
#           \.                                #   label dot
#       )+                                    # one or more labels
#       [a-z]{2,63}                           # TLD (2-63 letters)
#     | localhost                             # or 'localhost'
#     | (?:\d{1,3}\.){3}\d{1,3}               # or IPv4 address
#   )
#   (?::\d{1,5})?                             # optional port
#   (?:/[^\s?#]*)?                            # optional path (no spaces, ?, #)
#   (?:\?[^\s#]*)?                            # optional query (no spaces, #)
#   (?:#[^\s]*)?                              # optional fragment (no spaces)
#   $                                         # end of string
#
# The pattern is case-insensitive to allow mixed-case hostnames.
_HTTP_URL_RE = re.compile(
    r'^https?://'
    r'(?:'
    r'(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}'
    r'|'
    r'localhost'
    r'|'
    r'(?:\d{1,3}\.){3}\d{1,3}'
    r')'
    r'(?::\d{1,5})?'
    r'(?:/[^\s?#]*)?'
    r'(?:\?[^\s#]*)?'
    r'(?:#[^\s]*)?'
    r'$',
    re.IGNORECASE,
)


def verify_path_format(address: str) -> bool:
    """
    Verify whether the given string is a valid HTTP or HTTPS URL.

    The check is performed using a precompiled regular expression that covers:
      - http:// or https:// scheme
      - domain names, 'localhost', or IPv4 addresses
      - optional port, path, query, and fragment
    The function is defensive and will return False for non-string inputs or
    on any unexpected error.

    Parameters:
        address (str): The input string to validate as an HTTP/HTTPS URL.

    Returns:
        bool: True if the given string matches the expected URL format;
              False otherwise.
    """
    try:
        if not isinstance(address, str):
            return False

        # Use .match with ^/$ anchors in the pattern to ensure full-string match.
        return _HTTP_URL_RE.match(address) is not None
    except Exception:
        # Never raise; return False on any unexpected failure.
        return False
