import re

# Precompiled regex to validate an email and capture its domain part.
# Using re.VERBOSE to allow inline comments for clarity.
_EMAIL_DOMAIN_RE = re.compile(
    r"""
    ^                          # Start of string
    [^\s@]+                    # Local part: one or more non-space, non-@ characters
    @                          # Literal '@' separating local part and domain
    (?P<domain>                # Start capturing group 'domain'
      (?:                      # One or more domain labels followed by a dot:
        [A-Za-z0-9]                # - label starts with an alphanumeric
        (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?  # - may contain hyphens, but not start/end with hyphen (1-63 chars)
        \.                         # - dot between labels
      )+
      [A-Za-z]{2,63}           # Final TLD: letters only, length between 2 and 63
    )                          # End capturing group 'domain'
    $                          # End of string
    """,
    re.VERBOSE,
)


def fetch_email_domain(mail_id: str) -> str | None:
    """
    Return the domain portion of a well-formed email address.

    Parameters:
        mail_id (str): The email address that needs to be parsed.

    Returns:
        str | None: The domain (text after '@') if the email is valid; otherwise None.

    Notes:
        This function uses guard clauses (early returns) and does not raise exceptions.
        Invalid or non-string input results in None.
    """
    # Guard clause: input must be a string
    if not isinstance(mail_id, str):
        return None

    # Attempt to match the full email pattern and extract the named 'domain' group
    match = _EMAIL_DOMAIN_RE.fullmatch(mail_id)
    if not match:
        return None  # Guard clause: invalid email format

    # Successful match: return the captured domain part
    return match.group('domain')
