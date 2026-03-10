import re

# Pre-compile the regex for performance and clarity.
# This pattern ensures:
# - Local part: one or more non-whitespace, non-@ characters.
# - '@' separator.
# - Domain: labels of alphanumerics/hyphens separated by dots, requiring at least one dot.
_EMAIL_DOMAIN_REGEX = re.compile(r'^[^@\s]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)$')


def retrieve_domain_from_email(email_address: str):
    """
    Retrieve the domain from a valid email address using a regular expression.

    Args:
        email_address (str): The email address to process and analyze.

    Returns:
        str | None: The domain portion if the email is valid; otherwise, None.
    """
    try:
        if not isinstance(email_address, str):
            return None

        # Trim whitespace around the email address.
        email_address = email_address.strip()
        if not email_address:
            return None

        match = _EMAIL_DOMAIN_REGEX.match(email_address)
        if not match:
            return None

        return match.group('domain')
    except Exception:
        # Do not raise exceptions; return None on any unexpected error.
        return None
