import re

# Precompiled regex to validate the email and capture the domain.
# - Local part allows common characters and dot-separated atoms.
# - Domain part enforces valid DNS label rules and requires at least one dot and a 2+ char TLD (alnum to allow punycode).
_EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z0-9]{2,})$"
)


def fetch_email_domain(address: str):
    """
    Retrieve the domain from a valid email address.
    - address: str, the email address to process and analyze.
    Returns:
      - The domain as a string if the email is valid.
      - None if the email is invalid.
    This function will not raise exceptions.
    """
    try:
        if not isinstance(address, str):
            return None

        trimmed = address.strip()
        if not trimmed:
            return None

        match = _EMAIL_REGEX.match(trimmed)
        if not match:
            return None

        return match.group("domain")
    except Exception:
        # Ensure no exceptions escape the function.
        return None
