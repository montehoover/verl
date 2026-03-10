import re

# Compiled regex to validate email structure and capture the domain part.
# Local-part: common RFC-like characters.
# Domain: one or more labels separated by dots, ending with a TLD of at least 2 letters.
_EMAIL_DOMAIN_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})$"
)

def get_email_domain(addr: str):
    """
    Extract the domain portion of an email address using a regular expression.

    Args:
        addr (str): The email address to examine.

    Returns:
        str | None: If valid, the domain part (after '@'); otherwise None.

    Notes:
        - The function will not raise exceptions; it returns None on any error.
    """
    try:
        if not isinstance(addr, str):
            return None

        s = addr.strip()
        if not s:
            return None

        m = _EMAIL_DOMAIN_RE.fullmatch(s)
        if not m:
            return None

        return m.group(1)
    except Exception:
        # Explicitly ensure no exceptions escape the function.
        return None
