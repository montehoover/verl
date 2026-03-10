import re

# Pre-compiled regex for basic email validation and domain capture.
# Local-part: dot-separated atoms of allowed characters (no leading/trailing/consecutive dots).
# Domain: labels separated by dots; labels don't start/end with hyphen.
_LOCAL_ATOM = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
_LOCAL_PART = rf"{_LOCAL_ATOM}(?:\.{_LOCAL_ATOM})?"
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN = rf"{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})*"
_EMAIL_REGEX = re.compile(rf"^{_LOCAL_PART}@({_DOMAIN})$", re.IGNORECASE)


def retrieve_domain_from_email(email_address: str):
    """
    Extract the domain from a valid email address using a regular expression.

    Args:
        email_address (str): The email address to process and analyze.

    Returns:
        str | None: The domain string if the email address is valid; otherwise None.
    """
    try:
        if not isinstance(email_address, str):
            return None

        candidate = email_address.strip()
        match = _EMAIL_REGEX.fullmatch(candidate)
        if not match:
            return None
        return match.group(1)
    except Exception:
        # Ensure no exceptions propagate as per requirements.
        return None
