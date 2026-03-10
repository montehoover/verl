import re

def is_valid_email(email: str) -> bool:
    """
    Checks if a given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # A basic regex for email validation.
    # This regex is not exhaustive but covers most common cases.
    # For a more robust solution, a dedicated library might be better.
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    return False

def split_email_components(email: str) -> dict[str, str | None]:
    """
    Splits an email address into its username and domain components.

    Args:
        email: The email address string.

    Returns:
        A dictionary with 'username' and 'domain' keys.
        If the email does not contain an '@' symbol, the 'username' will be
        the original email string and 'domain' will be None.
    """
    parts = email.split('@', 1)
    if len(parts) == 2:
        return {'username': parts[0], 'domain': parts[1]}
    else:
        # Handle cases where there is no '@' symbol or it's malformed
        # for splitting purposes, though is_valid_email should catch most.
        return {'username': parts[0], 'domain': None}

def fetch_email_domain(email: str) -> str | None:
    """
    Extracts the domain from a valid email address using a regular expression.

    Args:
        email: The email address string.

    Returns:
        The domain as a string if the email is valid and domain is found,
        otherwise None.
    """
    if not is_valid_email(email):
        return None
    
    # Regex to capture the domain part of an email address.
    # It looks for characters after '@' that form a valid domain.
    domain_pattern = r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    match = re.search(domain_pattern, email)
    
    if match:
        # The domain includes the '@' symbol from the search, so slice it off.
        return match.group(0)[1:]
    return None
