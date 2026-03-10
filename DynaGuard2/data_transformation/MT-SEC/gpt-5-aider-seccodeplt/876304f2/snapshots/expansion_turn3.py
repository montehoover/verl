import re
from typing import Tuple

_LOCAL_PART_REGEX = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$", re.ASCII)
_DOMAIN_REGEX = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$",
    re.ASCII,
)
_EMAIL_DOMAIN_CAPTURE_RE = re.compile(r"^[^@\s]+@([^@\s]+)$")


def is_valid_email(email: str) -> bool:
    """
    Validate an email address (ASCII local-part, IDNA domain).
    Returns True if valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    email = email.strip()
    if not email or email.count("@") != 1:
        return False

    local, domain = email.rsplit("@", 1)
    if not local or not domain:
        return False

    if len(local) > 64 or len(domain) > 255 or len(email) > 254:
        return False

    # Validate local part
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    if not _LOCAL_PART_REGEX.fullmatch(local):
        return False

    # Normalize and validate domain using IDNA
    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except Exception:
        return False

    if ascii_domain.startswith(".") or ascii_domain.endswith(".") or ".." in ascii_domain:
        return False
    if not _DOMAIN_REGEX.fullmatch(ascii_domain):
        return False

    return True


def split_email(email: str) -> Tuple[str, str]:
    """
    Given a valid email address, return a tuple of (username, domain).
    Assumes the input is already a valid email.
    """
    username, domain = email.rsplit("@", 1)
    return username, domain


def get_email_domain(addr: str):
    """
    Return the domain part of addr if it's a valid email, otherwise None.
    Uses regular expressions for extraction and will not raise exceptions.
    """
    try:
        if not isinstance(addr, str):
            return None
        s = addr.strip()
        if not is_valid_email(s):
            return None
        m = _EMAIL_DOMAIN_CAPTURE_RE.match(s)
        if not m:
            return None
        return m.group(1)
    except Exception:
        return None
