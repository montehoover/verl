import re

# Basic, practical email validation regex.
# - Local part allows common unquoted characters and dot-separated segments.
# - Domain part enforces DNS-like labels and a TLD of at least 2 letters.
_LOCAL_PART = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
_DOMAIN = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
_EMAIL_REGEX = re.compile(rf"^{_LOCAL_PART}@{_DOMAIN}$", re.IGNORECASE)


def is_valid_email(email: str) -> bool:
    """
    Return True if the given string is a valid-looking email address, else False.
    This is a practical validator and does not attempt full RFC compliance.
    """
    if not isinstance(email, str):
        return False
    # Common practical length constraints
    if not (3 <= len(email) <= 254):
        return False
    return _EMAIL_REGEX.fullmatch(email) is not None
