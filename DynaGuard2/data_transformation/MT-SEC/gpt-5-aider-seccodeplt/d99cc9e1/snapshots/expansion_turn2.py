from typing import Final, Tuple


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
