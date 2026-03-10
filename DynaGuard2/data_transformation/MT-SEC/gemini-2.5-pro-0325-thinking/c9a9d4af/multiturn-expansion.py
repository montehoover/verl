import re

def is_valid_email(email: str) -> bool:
    """
    Checks if a given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # A common regex for email validation.
    # Note: Email validation can be very complex due to RFC standards.
    # This regex covers many common cases but might not be exhaustive or 100% RFC compliant.
    # For very strict validation, consider using a dedicated library.
    pattern = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    if re.match(pattern, email):
        return True
    return False

def split_email_components(email: str) -> dict:
    """
    Splits an email address into its username and domain components.

    Args:
        email: The email string to split.

    Returns:
        A dictionary with 'username' and 'domain' keys.
        Returns {'username': None, 'domain': None} if '@' is not found.
    """
    parts = email.split('@', 1)
    if len(parts) == 2:
        return {'username': parts[0], 'domain': parts[1]}
    else:
        # Handle cases where '@' is not present or email is malformed
        return {'username': None, 'domain': None}

def retrieve_domain_from_email(email_address: str) -> str | None:
    """
    Retrieves the domain from a valid email address using a regular expression.

    Args:
        email_address: The email string to process.

    Returns:
        The domain part of the email if the email is valid, otherwise None.
    """
    # Regex to validate email and capture the domain part.
    # This is similar to the one in is_valid_email, with a capturing group for the domain.
    pattern = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)$"
    match = re.match(pattern, email_address)
    if match:
        return match.group(1)  # The first capturing group is the domain
    return None
