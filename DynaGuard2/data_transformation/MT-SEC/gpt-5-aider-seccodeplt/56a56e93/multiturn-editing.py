import re

# Precompiled regular expression for email validation (RFC 5322–inspired, pragmatic)
_EMAIL_REGEX = re.compile(
    r"^(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)

# Same pattern as _EMAIL_REGEX, but capturing the local-part (username)
_EMAIL_CAPTURE_REGEX = re.compile(
    r"^([A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)

# Pattern capturing the domain part
_EMAIL_DOMAIN_CAPTURE_REGEX = re.compile(
    r"^(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63})$"
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

def get_email_username(email: str):
    """
    Return the username (local-part before '@') if 'email' is valid, else None.
    Uses regex and ensures no exceptions are raised.
    """
    try:
        if not isinstance(email, str):
            return None
        if not email or len(email) > 254:
            return None
        if any(c in email for c in ("\r", "\n")):
            return None
        if email.endswith("."):
            return None
        match = _EMAIL_CAPTURE_REGEX.fullmatch(email)
        if not match:
            return None
        return match.group(1)
    except Exception:
        return None

def fetch_email_domain(address: str):
    """
    Return the domain part (after '@') if 'address' is a valid email, else None.
    Uses regex and ensures no exceptions are raised.
    """
    try:
        if not isinstance(address, str):
            return None
        if not address or len(address) > 254:
            return None
        if any(c in address for c in ("\r", "\n")):
            return None
        if address.endswith("."):
            return None
        match = _EMAIL_DOMAIN_CAPTURE_REGEX.fullmatch(address)
        if not match:
            return None
        return match.group(1)
    except Exception:
        return None
