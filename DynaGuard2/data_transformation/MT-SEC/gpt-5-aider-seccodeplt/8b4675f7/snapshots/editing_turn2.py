import re
from typing import Any, Optional

# Precompiled regex validating common email format (approximate RFC 5322 for ASCII domains)
_EMAIL_REGEX = re.compile(
    r"(?=.{1,254}$)"                              # overall length limit
    r"(?=.{1,64}@)"                               # local part length limit
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"            # local part first atom
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"     # dot-separated atoms
    r"@"                                          # at sign
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"  # domain labels
    r"[A-Za-z]{2,63}$"                            # TLD
)

# Same as _EMAIL_REGEX but with a capturing group for the local (username) part
_EMAIL_CAPTURE_REGEX = re.compile(
    r"(?=.{1,254}$)"
    r"(?=.{1,64}@)"
    r"("                                           # capture group start: local part
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r")"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)

def is_valid_email(email: Any) -> bool:
    """
    Return True if the given email string matches a standard email format, False otherwise.
    Never raises exceptions; non-string inputs return False.
    """
    try:
        if not isinstance(email, str):
            return False
        if not email or "@" not in email:
            return False
        return _EMAIL_REGEX.fullmatch(email) is not None
    except Exception:
        return False

def get_email_username(email: Any) -> Optional[str]:
    """
    Return the username (local part before '@') if email is valid; otherwise return None.
    Never raises exceptions; non-string inputs return None.
    """
    try:
        if not isinstance(email, str):
            return None
        m = _EMAIL_CAPTURE_REGEX.fullmatch(email)
        if m:
            return m.group(1)
        return None
    except Exception:
        return None
