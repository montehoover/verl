import re

# Precompiled regular expression to validate email and capture the domain.
# Rules:
# - Local part: allows common permitted characters (simplified)
# - Domain: one or more labels separated by dots
#   - Each label starts and ends with alphanumeric, may contain hyphens
#   - Final TLD is alphabetic and at least 2 characters
_EMAIL_DOMAIN_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,}))$"
)


def retrieve_email_domain(email_str: str):
    """
    Extract the domain portion from a well-formed email address using regex.

    Args:
        email_str (str): The email address to parse.

    Returns:
        str | None: The domain if the email is valid; otherwise, None.

    Notes:
        - This function does not raise exceptions.
        - Leading/trailing whitespace is ignored.
    """
    try:
        if not isinstance(email_str, str):
            return None

        candidate = email_str.strip()
        match = _EMAIL_DOMAIN_RE.match(candidate)
        if not match:
            return None

        return match.group(1)
    except Exception:
        # Ensure no exceptions are propagated as per requirements.
        return None
