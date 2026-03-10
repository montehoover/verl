import re

# Pre-compiled regular expression for a simplified common email format.
# This regex checks for:
# 1. One or more characters (not '@') before the '@' symbol (local part).
# 2. The '@' symbol.
# 3. The domain part, which consists of:
#    a. One or more characters (alphanumeric, hyphen) for the domain name.
#    b. A dot ('.').
#    c. Two or more alphabetic characters for the top-level domain (e.g., .com, .org).
# The domain part (item 3) is captured in a group.
_EMAIL_REGEX = re.compile(r"[^@]+@([^@]+\.[a-zA-Z]{2,})")


def _get_email_match(email_address: str) -> re.Match | None:
    """
    Validates an email string against a pre-defined regex and returns a match object.

    This function uses `re.fullmatch` to ensure the entire string conforms to
    the email pattern defined in `_EMAIL_REGEX`.

    Args:
        email_address: The email address string to validate.

    Returns:
        A `re.Match` object if the `email_address` is a valid format and
        fully matches the regex, otherwise `None`.
    """
    # Attempt to match the entire email_address string against the pre-compiled regex.
    return _EMAIL_REGEX.fullmatch(email_address)


def _extract_domain_from_match_object(email_match: re.Match) -> str:
    """
    Extracts the domain string from a `re.Match` object.

    This function assumes the `re.Match` object is the result of a successful
    match using `_EMAIL_REGEX`, where the domain is the first captured group.

    Args:
        email_match: A `re.Match` object resulting from a successful regex match
                     on an email string.

    Returns:
        The domain part of the email address as a string.
    """
    # The domain is captured in the first group (index 1) of the regex match.
    return email_match.group(1)


def retrieve_email_domain(email_str: str) -> str | None:
    """
    Extracts the domain portion from an email address string.

    This function orchestrates the validation of the email string format and
    the extraction of the domain. It returns the domain if the email is
    well-formed according to a simplified, common pattern; otherwise, it returns None.

    Args:
        email_str: The email address string to be parsed.

    Returns:
        The domain portion of the email address as a string if the input
        `email_str` is valid and a domain can be extracted.
        Returns `None` if `email_str` is not a string, is malformed, or
        does not match the expected email pattern.
    
    Raises:
        This function does not raise any exceptions.
    """
    # Guard clause: Ensure the input is a string.
    # If not, it cannot be a valid email address, so return None immediately.
    if not isinstance(email_str, str):
        return None
    
    # Attempt to get a regex match object for the given email string.
    # _get_email_match will return a match object if the email format is valid,
    # otherwise it returns None.
    match_object = _get_email_match(email_str)
    
    # Check if a valid match was found.
    if match_object:
        # If a match exists, extract the domain part from the match object.
        return _extract_domain_from_match_object(match_object)
    else:
        # If no match was found (i.e., email_str is not a valid format), return None.
        return None
