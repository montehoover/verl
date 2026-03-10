import re

# Precompiled regex for basic RFC 5322-compliant "dot-atom" local part and domain names
_EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,})$"
)


def is_valid_email(email: str) -> bool:
    """
    Return True if `email` is a syntactically valid email address; otherwise False.
    - Enforces overall length <= 254 and local part length <= 64.
    - Local part: dot-atom with allowed characters.
    - Domain: one or more labels separated by dots, labels are alnum with optional interior hyphens,
      and TLD is at least 2 letters.
    """
    if not isinstance(email, str):
        return False

    email = email.strip()

    # RFC 5321 limits: 254 total, 64 local-part
    if len(email) > 254:
        return False

    match = _EMAIL_REGEX.fullmatch(email)
    if not match:
        return False

    local = match.group("local")
    if len(local) > 64:
        return False

    return True
