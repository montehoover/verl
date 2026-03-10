import re
from typing import Final, Tuple, Optional


def check_keyword_in_string(text: str, keyword: str) -> bool:
    """
    Return True if `keyword` is present anywhere in `text`, otherwise False.

    Notes:
    - The check is case-sensitive.
    - An empty `keyword` returns False to avoid treating the empty string as present in any text.
    """
    if not isinstance(text, str) or not isinstance(keyword, str):
        raise TypeError("Both 'text' and 'keyword' must be strings.")

    if keyword == "":
        return False

    return keyword in text


def split_email_parts(email: str) -> Tuple[str, str]:
    """
    Split a single email address into (local_part, domain).

    Returns:
        A tuple (local_part, domain).

    Raises:
        TypeError: if `email` is not a string.
        ValueError: if `email` does not contain exactly one '@' character, or if
                    the local_part or domain is empty.
    """
    if not isinstance(email, str):
        raise TypeError("'email' must be a string.")

    stripped = email.strip()
    if stripped.count("@") != 1:
        raise ValueError("Email must contain exactly one '@' character.")

    local_part, domain = stripped.split("@", 1)

    if not local_part or not domain:
        raise ValueError("Email must have non-empty local part and domain.")

    return local_part, domain


# Regex for a commonly accepted email format (unquoted local part, domain with labels and TLD)
# Captures the domain in a named group 'domain'.
_EMAIL_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+(?:[A-Za-z]{2,}))$"
)


def retrieve_email_domain(email_str: str) -> Optional[str]:
    """
    Extract the domain from a well-formed email address using regular expressions.

    Returns:
        The domain string if the email is well-formed; otherwise, None.
    """
    if not isinstance(email_str, str):
        return None

    s = email_str.strip()
    match = _EMAIL_REGEX.fullmatch(s)
    if not match:
        return None

    return match.group("domain")
