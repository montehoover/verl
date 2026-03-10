import re

# Domain regex:
# - One or more labels separated by dots
#   - Each label starts and ends with alphanumeric, may contain hyphens in between
#   - Final TLD is alphabetic and at least 2 characters
_DOMAIN_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,})$"
)

# Simplified local-part regex (commonly permitted characters)
_LOCAL_PART_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$")


def _split_email(email: str):
    """
    Pure function to extract local and domain parts from an email-like string.

    Returns:
        tuple[str, str] | None: (local, domain) if a single '@' splits the string
        into non-empty parts; otherwise, None.
    """
    if not isinstance(email, str):
        return None

    candidate = email.strip()
    if candidate.count("@") != 1:
        return None

    local, domain = candidate.split("@", 1)
    if not local or not domain:
        return None

    return local, domain


def _is_valid_domain(domain: str) -> bool:
    """
    Pure function to validate a domain string.
    """
    if not isinstance(domain, str) or not domain:
        return False
    return _DOMAIN_RE.fullmatch(domain) is not None


def _is_valid_local_part(local: str) -> bool:
    """
    Pure function to validate an email local-part (simplified).
    """
    if not isinstance(local, str) or not local:
        return False
    return _LOCAL_PART_RE.fullmatch(local) is not None


def retrieve_email_domain(email_str: str):
    """
    Extract the domain portion from a well-formed email address using pure helpers.

    Args:
        email_str (str): The email address to parse.

    Returns:
        str | None: The domain if the email is valid; otherwise, None.

    Notes:
        - This function does not raise exceptions.
        - Leading/trailing whitespace is ignored.
    """
    try:
        parts = _split_email(email_str)
        if not parts:
            return None

        local, domain = parts

        if not _is_valid_local_part(local):
            return None

        if not _is_valid_domain(domain):
            return None

        return domain
    except Exception:
        # Ensure no exceptions are propagated as per requirements.
        return None
