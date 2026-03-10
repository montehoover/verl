import re

"""
Provides a function to validate if a string is a well-formed HTTP/HTTPS URL.

This module contains a regular expression for matching common HTTP/HTTPS URL
structures and a function `path_check` that uses this regex to validate
input strings.
"""

# Predefined regular expression for HTTP/HTTPS URLs.
# This pattern aims to validate common URL structures.
# It covers domain names, localhost, IPv4, and simplified IPv6 addresses,
# along with optional port, path, query, and fragment components.
# For production use, a more comprehensive and thoroughly tested regex might be needed,
# or a dedicated URL parsing library.
_URL_PATTERN = re.compile(
    r"^https?://"  # Scheme: http or https
    # Host: domain name, localhost, IPV4, or IPV6
    r"(?:"  # Start of host alternatives
    # Domain name:
    # e.g., example.com, sub.example.co.uk
    # Consists of labels separated by dots. Each label:
    # - starts with an alphanumeric character
    # - contains alphanumerics or hyphens (not at start/end)
    # - is 1-63 characters long
    # TLD is 2-63 characters (letters or can include numbers/hyphens for newer gTLDs/ccTLDs)
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,63}|[A-Z0-9-]{2,63}\.?))|"
    r"localhost|"  # localhost
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|"  # IP v4 address
    # IP v6 address (simplified, enclosed in square brackets)
    # A full IPv6 regex is very complex; this covers common cases.
    r"\[[A-F0-9:]+\]"
    r")"  # End of host alternatives
    r"(?::\d+)?"  # Optional port number (e.g., :8080)
    r"(?:/[^?#\s]*)?"  # Optional path (e.g., /path/to/resource)
    r"(?:\?[^#\s]*)?"  # Optional query string (e.g., ?key=value&key2=value2)
    r"(?:#\S*)?$",  # Optional fragment (e.g., #section)
    re.IGNORECASE  # Perform case-insensitive matching
)


def path_check(u: str) -> bool:
    """
    Checks whether a given string represents a valid http or https URL
    based on a predefined regular expression.

    Args:
        u: The string input to be validated as a URL.

    Returns:
        True if the input string is a valid URL according to the regex, 
        False otherwise. This function does not raise exceptions for 
        invalid input types (e.g., non-string inputs will return False).
    """
    if not isinstance(u, str):
        return False  # Ensure no TypeError for non-string inputs
    
    # re.fullmatch ensures the entire string matches the pattern
    match = _URL_PATTERN.fullmatch(u)
    return bool(match)
