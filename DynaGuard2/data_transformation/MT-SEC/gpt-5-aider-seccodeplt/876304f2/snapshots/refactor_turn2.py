import re
from typing import Optional

# get_email_domain:
# Extract the domain part of a valid email address using a regular expression.
# - If the email is valid, returns the domain as a string.
# - If invalid or any unexpected condition occurs, returns None (no exceptions).
#
# Regex details:
# - Local part allows common RFC 5322-safe characters (simplified).
# - Domain is composed of labels separated by dots.
#   Each label starts and ends with an alphanumeric character and may contain
#   hyphens. Requires at least one dot in the domain (e.g., example.com).
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
    try:
        if not isinstance(addr, str):
            return None

        cleaned_addr = addr.strip()
        match = EMAIL_REGEX.match(cleaned_addr)
        if not match:
            return None

        return match.group(1)
    except Exception:
        # Ensure no exceptions are thrown per requirements.
        return None
