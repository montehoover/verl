import re
from typing import Optional

EMAIL_DOMAIN_REGEX = re.compile(
    r'^[^@\s]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)$'
)


def retrieve_domain_from_email(email_address: str) -> Optional[str]:
    """
    Retrieve the domain from a valid email address using a regular expression.

    Args:
        email_address: The email address to process and analyze.

    Returns:
        The domain portion if the email is valid; otherwise, None.
    """
    if not isinstance(email_address, str):
        return None

    email_address = email_address.strip()
    if not email_address:
        return None

    match = EMAIL_DOMAIN_REGEX.match(email_address)
    if not match:
        return None

    return match.group('domain')
