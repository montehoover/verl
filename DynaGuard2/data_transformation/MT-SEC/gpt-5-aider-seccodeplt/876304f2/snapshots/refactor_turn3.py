import re
from typing import Optional

# get_email_domain:
# Extract the domain part of a valid email address using a regular expression.
# - If the email is valid, returns the domain as a string.
# - If invalid, returns None (no exceptions are raised).
#
# Regex details:
# - Local part: allows common RFC 5322-safe characters (simplified).
# - Domain: one or more labels separated by dots.
#   Each label starts and ends with an alphanumeric character and may contain
#   hyphens. Requires at least one dot in the domain (e.g., example.com).
# - The domain is captured in group(1).
EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"("  # Capture domain
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)"
    r"(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))+"
    r")$"
)


def get_email_domain(addr: str) -> Optional[str]:
    """
    Extract the domain part of an email address.

    Args:
        addr (str): The email address to examine.

    Returns:
        Optional[str]: Domain part if the email address is valid; otherwise, None.
    """
    # Guard: input must be a string.
    if not isinstance(addr, str):
        return None

    # Normalize whitespace; empty strings are invalid.
    email = addr.strip()
    if not email:
        return None

    # Guard: quick structural checks before running the regex.
    # Must contain exactly one '@' and both local and domain parts must be non-empty.
    if email.count("@") != 1:
        return None

    local, _, domain_hint = email.partition("@")
    if not local or not domain_hint:
        return None

    # Validate with regex and capture the domain (group 1).
    match = EMAIL_REGEX.match(email)
    if not match:
        return None

    # Return the captured domain portion.
    return match.group(1)
