import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an email address or an FTP URL using regular expressions.

    Args:
        input_path: The string to be validated.

    Returns:
        True if the input is a valid email format or FTP URL format, False otherwise.
    """
    # Regex for email validation
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    # Regex for FTP URL validation (simplified)
    # ftp://[user:password@]host[:port]/path
    ftp_regex = r"^ftp://([a-zA-Z0-9._%+-]+(:[a-zA-Z0-9._%+-]+)?@)?([a-zA-Z0-9.-]+)(:[0-9]+)?(/.*)?$"

    if re.fullmatch(email_regex, input_path) or re.fullmatch(ftp_regex, input_path):
        return True
    else:
        return False
