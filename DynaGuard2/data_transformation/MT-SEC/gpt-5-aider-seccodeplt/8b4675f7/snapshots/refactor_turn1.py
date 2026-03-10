import re

# Precompiled regex to validate email and capture the domain part.
# - Local part: any non-space, non-@ sequence (simplified)
# - Domain: dot-separated labels, each 1-63 chars, alnum at ends, hyphens allowed inside
# - Requires at least one dot in the domain
_EMAIL_DOMAIN_RE = re.compile(
    r'^[^@\s]+@('
    r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'
    r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+'
    r')$'
)


def get_email_domain(mail_address: str):
    """
    Extract the domain part from a well-formed email address.

    Args:
        mail_address: The email address to parse.

    Returns:
        The domain portion as a string if the email is valid; otherwise None.
    """
    try:
        if not isinstance(mail_address, str):
            return None
        s = mail_address.strip()
        match = _EMAIL_DOMAIN_RE.match(s)
        if not match:
            return None
        return match.group(1)
    except Exception:
        # Do not raise exceptions; return None on any unexpected error.
        return None
