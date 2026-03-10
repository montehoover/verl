import re
from typing import Optional

# Pre-compiled regular expression for validating an email address and capturing the domain.
# This pattern allows a conventional local-part and either:
# - A domain name with labels (letters/digits/hyphens) and at least one dot, or
# - A domain-literal in square brackets (IPv4 or IPv6-like).
_EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)"
    r"|\[(?:IPv6:[A-Fa-f0-9:.]+|(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})\])$"
)


def extract_email_domain(email: str) -> Optional[str]:
    """
    Extract the domain part from an email address if it's valid.

    Args:
        email: The input email address to be processed.

    Returns:
        The domain part (everything after '@') if the input is a valid email address; otherwise, None.

    Notes:
        - This function does not raise exceptions; it returns None on invalid input.
        - Leading/trailing whitespace is ignored.
    """
    if not isinstance(email, str):
        return None

    candidate = email.strip()
    if not candidate:
        return None

    match = _EMAIL_REGEX.fullmatch(candidate)
    if not match:
        return None

    return match.group("domain")
