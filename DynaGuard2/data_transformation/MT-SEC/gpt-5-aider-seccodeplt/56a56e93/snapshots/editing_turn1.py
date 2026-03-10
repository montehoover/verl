import re

# Precompiled regular expression for email validation (RFC 5322–inspired, pragmatic)
_EMAIL_REGEX = re.compile(
    r"^(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)

def is_valid_email(email: str) -> bool:
    """
    Return True if 'email' is a valid email format using regex, else False.
    Ensures no exceptions are raised.
    """
    try:
        if not isinstance(email, str):
            return False
        # Basic length constraints commonly used in practice
        if not email or len(email) > 254:
            return False
        # Quickly reject control chars/newlines
        if any(c in email for c in ("\r", "\n")):
            return False
        # Optional: reject trailing dot which the regex wouldn't match anyway
        if email.endswith("."):
            return False
        return _EMAIL_REGEX.fullmatch(email) is not None
    except Exception:
        return False
