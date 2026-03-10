import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an email address or an FTP URL using regular expressions.
    
    Args:
        input_path: The email address or FTP URL to be validated
        
    Returns:
        True if the input is a valid email format or FTP URL format, False otherwise
    """
    # Regular expression pattern for email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Regular expression pattern for FTP URL validation
    ftp_pattern = r'^ftp://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?::\d+)?(?:/[^/\s]*)*/?$'
    
    # Check if the input matches either the email pattern or FTP URL pattern
    return bool(re.match(email_pattern, input_path) or re.match(ftp_pattern, input_path))
