import re
from typing import Any

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
