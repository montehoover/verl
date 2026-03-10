import re

# Precompiled regex to validate email and capture its domain part.
# - exactly one '@'
# - local-part: any non-whitespace, non-@ characters
# - domain: dot-separated labels; labels are alnum, may contain hyphens but not start/end with hyphen
# - final TLD: letters only, length 2-63
_EMAIL_DOMAIN_RE = re.compile(
    r'^[^\s@]+@'
    r'(?P<domain>('
    r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+'
    r'(?:[A-Za-z]{2,63})'
    r'))$'
)

def fetch_email_domain(mail_id: str):
    """
    Extract the domain portion from an email address.

    Args:
        mail_id: str - The email address that needs to be parsed.

    Returns:
        str | None: The domain portion if the input is a valid email, otherwise None.
    """
    if not isinstance(mail_id, str):
        return None

    match = _EMAIL_DOMAIN_RE.fullmatch(mail_id)
    if not match:
        return None
    return match.group('domain')
