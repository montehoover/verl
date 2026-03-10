import re

def fetch_email_domain(mail_id):
    """
    Extract the domain part from an email address.

    Args:
        mail_id (str): The email address to parse.

    Returns:
        str | None: The domain part (lowercased) if the email is valid, otherwise None.
    """
    if not isinstance(mail_id, str):
        return None

    email = mail_id.strip()
    if not email:
        return None

    # Pragmatic RFC-inspired regex:
    # - Local part: commonly allowed unquoted ASCII characters
    # - Domain: labels 1-63 chars, alnum plus hyphen (no leading/trailing hyphen), at least one dot
    pattern = (
        r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@("
        r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
        r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+)$"
    )

    match = re.fullmatch(pattern, email, flags=re.IGNORECASE)
    if not match:
        return None

    domain = match.group(1)
    return domain.lower()
