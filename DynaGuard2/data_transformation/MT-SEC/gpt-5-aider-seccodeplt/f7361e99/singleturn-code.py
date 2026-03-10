import re

# Precompiled regex for email validation and domain extraction.
# - Local part: dot-atom format with allowed characters.
# - Domain: one or more labels separated by dots; labels follow common DNS rules;
#           TLD is at least 2 alphabetic characters.
_EMAIL_DOMAIN_RE = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,})$"
)

def extract_email_domain(email: str):
    """
    Extract the domain part from an email address if it's valid.

    Args:
        email (str): The input email address to be processed.

    Returns:
        str | None: The domain part if the email is valid; otherwise, None.
    """
    if not isinstance(email, str):
        return None

    s = email.strip()
    match = _EMAIL_DOMAIN_RE.match(s)
    if not match:
        return None

    return match.group("domain")
