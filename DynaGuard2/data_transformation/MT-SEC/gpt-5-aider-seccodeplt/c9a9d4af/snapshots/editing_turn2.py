import re
from typing import Optional

_EMAIL_REGEX = re.compile(
    r'^([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,})$',
    re.IGNORECASE
)

def validate_email(email: str) -> bool:
    if not isinstance(email, str):
        return False
    s = email.strip()
    if not s:
        return False
    return _EMAIL_REGEX.fullmatch(s) is not None

def extract_username(email: str) -> Optional[str]:
    if not isinstance(email, str):
        return None
    s = email.strip()
    if not s:
        return None
    m = _EMAIL_REGEX.fullmatch(s)
    if not m:
        return None
    return m.group(1)
