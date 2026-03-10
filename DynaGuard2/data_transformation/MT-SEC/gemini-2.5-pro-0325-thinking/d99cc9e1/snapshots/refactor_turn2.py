import re

# Regex for a simplified common email format.
# It looks for one or more characters (not '@') before the '@' symbol,
# then the '@' symbol, then the domain part.
# The domain part consists of one or more characters (alphanumeric, hyphen)
# followed by a dot, and then two or more alphabetic characters (e.g., .com, .org).
_EMAIL_REGEX = re.compile(r"[^@]+@([^@]+\.[a-zA-Z]{2,})")

def _get_email_match(email_str: str) -> re.Match | None:
    """
    Validates the email string format and returns a match object if valid.

    Args:
        email_str: The email address string.

    Returns:
        A re.Match object if the email_str matches the expected format, None otherwise.
    """
    return _EMAIL_REGEX.fullmatch(email_str)

def _extract_domain_from_match_object(match: re.Match) -> str:
    """
    Extracts the domain from a regex match object.

    Args:
        match: A re.Match object from a successful email regex match.

    Returns:
        The domain string (captured group 1).
    """
    # The domain is in the first capturing group
    return match.group(1)

def retrieve_email_domain(email_str: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        email_str: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
    
    Raises:
        The function should not raise any exceptions.
    """
    if not isinstance(email_str, str):
        return None
    
    match = _get_email_match(email_str)
    
    if match:
        return _extract_domain_from_match_object(match)
    else:
        return None
