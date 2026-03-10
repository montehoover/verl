import re

def check_path_validity(path_string: str) -> bool:
    """
    Verifies if a given string is a valid email address.

    Args:
        path_string: The string to validate as an email address.

    Returns:
        True if the email is formatted correctly, False otherwise.
    """
    # Regular expression for validating an email
    # This is a common regex, but email validation can be complex.
    # For production systems, consider using a well-vetted library.
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(email_regex, path_string):
        return True
    else:
        return False
