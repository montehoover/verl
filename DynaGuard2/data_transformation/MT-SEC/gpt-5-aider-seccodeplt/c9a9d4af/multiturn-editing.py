import re

_EMAIL_REGEX = re.compile(
    r'^[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})$',
    re.IGNORECASE
)

def retrieve_domain_from_email(email_address: str):
    if not isinstance(email_address, str):
        return None
    s = email_address.strip()
    if not s:
        return None
    m = _EMAIL_REGEX.fullmatch(s)
    if not m:
        return None
    return m.group(1)
