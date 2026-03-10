import re

_EMAIL_REGEX = re.compile(
    r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$',
    re.IGNORECASE
)

def validate_email(email: str) -> bool:
    if not isinstance(email, str):
        return False
    s = email.strip()
    if not s:
        return False
    return _EMAIL_REGEX.fullmatch(s) is not None
