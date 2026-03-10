import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an email address using regular expressions.
    
    Args:
        input_path: The email address to be validated
        
    Returns:
        True if the input is a valid email format, False otherwise
    """
    # Regular expression pattern for email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the input matches the email pattern
    return bool(re.match(email_pattern, input_path))
