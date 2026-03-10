import re

def check_path_validity(path_string: str) -> bool:
    """
    Verify if a given string is a valid email address or FTP URL using regular expressions.
    
    Args:
        path_string: The string to validate as an email address or FTP URL
        
    Returns:
        True if the string is a valid email or FTP URL, False otherwise
    """
    # Regular expression pattern for email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Regular expression pattern for FTP URL validation
    ftp_pattern = r'^ftp://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
    
    # Check if the string matches either the email pattern or FTP pattern
    return bool(re.match(email_pattern, path_string) or re.match(ftp_pattern, path_string))
