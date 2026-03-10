import re

# Precompile the email validation regex pattern with a capturing group for the domain
EMAIL_REGEX = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,})$"
)

def get_email_domain(addr):
    """
    Extract the domain part from a valid email address.
    Returns the domain as a string if valid, otherwise None.
    """
    if not isinstance(addr, str):
        return None
    match = EMAIL_REGEX.match(addr)
    if not match:
        return None
    return match.group(1)
