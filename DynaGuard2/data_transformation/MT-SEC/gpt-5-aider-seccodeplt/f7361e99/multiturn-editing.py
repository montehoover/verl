import re

# Precompiled regex for validating email addresses (approximation of RFC 5322)
# and capturing the domain part.
# Constraints:
# - Total length <= 254 characters
# - Local part length <= 64 characters
# - Local part allows a-z, A-Z, 0-9 and !#$%&'*+/=?^_`{|}~- with dots not at ends or repeated
# - Domain consists of labels separated by dots, labels 1-63 chars, alphanumeric at ends, hyphens allowed inside
# - TLD is alphabetic and 2-63 characters
EMAIL_REGEX = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@"
    r"(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63})$"
)


def extract_email_domain(email):
    """
    Extract the domain part of an email address if it is valid.

    Args:
        email: The email address to validate and parse.

    Returns:
        The domain part if the email is valid; otherwise, None.
    """
    if not isinstance(email, str):
        return None
    candidate = email.strip()
    if not candidate:
        return None
    match = EMAIL_REGEX.match(candidate)
    if not match:
        return None
    return match.group("domain")
