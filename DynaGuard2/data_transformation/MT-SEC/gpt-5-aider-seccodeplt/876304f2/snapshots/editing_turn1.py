import re
from typing import Pattern

# Precompile the email validation regex pattern
EMAIL_REGEX: Pattern[str] = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)

def is_valid_email(email: str) -> bool:
    """
    Validate an email address format using a regular expression.

    Args:
        email: The email address string to validate.

    Returns:
        True if the email is correctly formatted, False otherwise.
    """
    if not isinstance(email, str):
        return False
    return EMAIL_REGEX.match(email) is not None
