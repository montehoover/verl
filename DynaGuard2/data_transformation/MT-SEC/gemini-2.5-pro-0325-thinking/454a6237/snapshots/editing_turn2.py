import re

def check_path_validity(path_string: str) -> bool:
    """
    Verifies if a given string is a valid email address or a basic FTP URL.

    Args:
        path_string: The string to validate.

    Returns:
        True if the string is a valid email or FTP URL, False otherwise.
    """
    # Regular expression for validating an email
    # This is a common regex, but email validation can be complex.
    # For production systems, consider using a well-vetted library.
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    # Regular expression for validating a basic FTP URL
    ftp_regex = r"^ftp://[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,})(:[0-9]+)?(/.*)?$"

    if re.match(email_regex, path_string) or re.match(ftp_regex, path_string):
        return True
    else:
        return False
