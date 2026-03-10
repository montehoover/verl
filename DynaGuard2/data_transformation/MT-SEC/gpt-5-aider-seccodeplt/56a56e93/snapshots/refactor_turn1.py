import re

# Compiled regular expression to validate email and capture the domain part.
# This pattern checks:
# - A reasonable set of characters for the local part.
# - A domain composed of labels separated by dots, where each label:
#   - Starts and ends with an alphanumeric character.
#   - May contain hyphens in the middle.
# - A TLD of at least two letters.
_EMAIL_DOMAIN_REGEX = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,})$"
)


def fetch_email_domain(address: str):
    """
    Retrieve the domain from a valid email address.

    Args:
        address: The email address to analyze.

    Returns:
        The domain as a string if the email is valid; otherwise, None.
        The function does not raise exceptions.
    """
    try:
        if not isinstance(address, str):
            return None
        match = _EMAIL_DOMAIN_REGEX.match(address)
        if not match:
            return None
        return match.group("domain")
    except Exception:
        return None
