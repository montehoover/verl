import re

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN_PATTERN = rf"(?:{_LABEL}\.)+{_LABEL}"
_EMAIL_RE = re.compile(
    rf"^[^@\s]+@(?P<domain>{_DOMAIN_PATTERN})$",
    re.IGNORECASE,
)

def fetch_email_domain(mail_id: str):
    """
    Extract the domain part from an email address (e.g., 'example.com' from 'user@example.com').
    Returns None if the email address is not well-formed.
    """
    if not isinstance(mail_id, str):
        return None

    mail_id = mail_id.strip()
    match = _EMAIL_RE.match(mail_id)
    if not match:
        return None

    return match.group("domain").lower()
