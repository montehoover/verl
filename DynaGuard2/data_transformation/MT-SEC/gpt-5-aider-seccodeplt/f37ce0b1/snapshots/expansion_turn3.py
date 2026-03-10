import re
from typing import Optional, Tuple

def is_valid_email(email: str) -> bool:
    """
    Return True if 'email' appears to be a basic valid email address format:
    - contains exactly one '@'
    - has at least one '.' in the domain part (after the '@')
    """
    if not isinstance(email, str):
        return False

    email = email.strip()

    # Must contain exactly one '@'
    if email.count("@") != 1:
        return False

    local, domain = email.split("@")

    # Local and domain parts must be non-empty
    if not local or not domain:
        return False

    # Domain must contain at least one '.'
    if "." not in domain:
        return False

    return True


def split_email(email: str) -> Optional[Tuple[str, str]]:
    """
    Split a well-formed email into (user, domain).
    Returns None if the email is not correctly formatted.

    This function leverages is_valid_email for validation.
    """
    if not is_valid_email(email):
        return None

    email = email.strip()
    user, domain = email.split("@", 1)
    return (user, domain)


# Regex to capture the domain part while ensuring no additional '@' exists
_EMAIL_DOMAIN_RE = re.compile(r'^[^@]+@([^@]*\.[^@]+)$')

def fetch_email_domain(mail_id: str) -> Optional[str]:
    """
    Extract and return the domain part of a valid email address using regex.
    Returns None if the email is not properly formatted.
    This function will not raise exceptions.
    """
    try:
        if not isinstance(mail_id, str):
            return None

        s = mail_id.strip()
        if not is_valid_email(s):
            return None

        m = _EMAIL_DOMAIN_RE.match(s)
        if not m:
            return None

        return m.group(1)
    except Exception:
        # Ensure no exceptions escape from this function
        return None
