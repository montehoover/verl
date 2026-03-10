import re
from typing import Pattern

# Precompiled regex for validating email addresses (approximation of RFC 5322)
# Constraints:
# - Total length <= 254 characters
# - Local part length <= 64 characters
# - Local part allows a-z, A-Z, 0-9 and !#$%&'*+/=?^_`{|}~- with dots not at ends or repeated
# - Domain consists of labels separated by dots, labels 1-63 chars, alphanumeric at ends, hyphens allowed inside
# - TLD is alphabetic and 2-63 characters
EMAIL_REGEX: Pattern[str] = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)


def is_valid_email(email: str) -> bool:
    """
    Validate an email address using a regular expression.

    Args:
        email: The email address to validate.

    Returns:
        True if the email address matches a standard format, False otherwise.
    """
    if not isinstance(email, str):
        return False
    candidate = email.strip()
    if not candidate:
        return False
    return EMAIL_REGEX.match(candidate) is not None
