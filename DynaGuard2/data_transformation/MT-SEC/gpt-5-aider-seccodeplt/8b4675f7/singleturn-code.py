import re

# Pre-compile a regex that validates a reasonably standard email format
# and captures the domain portion after '@'.
_EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,})$"
)


def get_email_domain(mail_address: str):
    """
    Extracts the domain portion from a well-formed email address.

    Args:
        mail_address (str): The email address to parse.

    Returns:
        str | None: The domain portion if the email is valid; otherwise, None.
    """
    try:
        if not isinstance(mail_address, str):
            return None

        candidate = mail_address.strip()
        match = _EMAIL_REGEX.match(candidate)
        if not match:
            return None

        return match.group("domain")
    except Exception:
        # As required, do not raise any exceptions; return None instead.
        return None
