"""Utilities for validating email addresses and extracting domains.

This module exposes a single helper function, `extract_email_domain`,
which validates an email address with a regular expression and returns
its domain part (the substring after the '@' character).
"""
import re
from typing import Optional

# Pre-compiled regular expression for validating an email address and capturing
# the domain part. The pattern captures two named groups:
# - 'local'  : the local-part before the '@'
# - 'domain' : the domain-part after the '@' (this is what we return)
#
# The domain portion supports:
# - Traditional DNS names composed of labels with letters/digits/hyphens,
#   requiring at least one dot (e.g., example.com, mail.example.co.uk).
# - Domain literals enclosed in square brackets for IPv4 and IPv6 forms
#   (e.g., [192.0.2.1], [IPv6:2001:db8::1]).
_EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)"
    r"|\[(?:IPv6:[A-Fa-f0-9:.]+|(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})\])$"
)


def extract_email_domain(email: str) -> Optional[str]:
    """Return the domain part of an email address if the address is valid.

    The function validates the input using a compiled regular expression and
    returns the substring after the '@' symbol (the domain) when the input
    resembles a conventional email address. If the input is not a string, is
    empty after trimming, or does not match the expected pattern, the function
    returns None.

    This function is designed to be safe to call: it does not raise exceptions.

    Args:
        email: The input email address to be processed.

    Returns:
        The domain part (everything after '@') when the email is valid.
        Otherwise, None.

    Examples:
        >>> extract_email_domain("user@example.com")
        'example.com'
        >>> extract_email_domain("USER+tag@sub.example.co.uk")
        'sub.example.co.uk'
        >>> extract_email_domain("not-an-email")
        >>> extract_email_domain("  user@example.com  ")
        'example.com'
    """
    # Ensure input is a string; return None for non-string inputs.
    if not isinstance(email, str):
        return None

    # Normalize whitespace by trimming leading/trailing spaces.
    normalized_email = email.strip()
    if not normalized_email:
        return None

    # Attempt to fully match the normalized input against the email regex.
    regex_match = _EMAIL_REGEX.fullmatch(normalized_email)
    if not regex_match:
        return None

    # Return the captured 'domain' part of the email address.
    return regex_match.group("domain")
