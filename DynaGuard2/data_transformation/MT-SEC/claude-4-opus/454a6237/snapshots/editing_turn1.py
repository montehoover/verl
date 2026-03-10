import re

def check_path_validity(path_string: str) -> bool:
    """
    Verify if a given string is a valid email address using regular expressions.
    
    Args:
        path_string: The string to validate as an email address
        
    Returns:
        True if the email is formatted correctly, False otherwise
    """
    # Regular expression pattern for email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the string matches the email pattern
    return bool(re.match(email_pattern, path_string))
