import re

# Precompiled regex for common email validation.
# - Total length up to 254 characters
# - Local part up to 64 characters
# - Local part allows dot-separated atoms of permitted characters
# - Domain is one or more labels separated by dots, labels can't start/end with hyphen
# - TLD is at least 2 letters or punycode (xn--)
EMAIL_REGEX = re.compile(
    r"""
    ^(?=.{1,254}$)                 # overall length constraint
    (?=.{1,64}@)                   # local part length constraint
    (?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+
       (?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*   # dot-separated atoms, no consecutive dots
    )
    @
    (?:
      (?:[A-Za-z0-9]               # domain label start
         (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?  # interior with hyphens, no start/end hyphen
      )
      \.
    )+
    (?:[A-Za-z]{2,}|xn--[A-Za-z0-9-]{2,})  # TLD
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def retrieve_email_domain(email_str: str) -> bool:
    """
    Validate whether the given string is a properly formatted email address.

    Args:
        email_str: The email address to validate.

    Returns:
        True if the email syntax matches common email patterns, False otherwise.
    """
    if not isinstance(email_str, str):
        return False
    if not email_str or email_str.count("@") != 1:
        return False
    return EMAIL_REGEX.fullmatch(email_str) is not None
