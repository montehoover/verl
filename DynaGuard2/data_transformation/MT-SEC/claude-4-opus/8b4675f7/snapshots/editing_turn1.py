import re

def is_valid_email(email):
    """
    Validates if an email address is valid according to standard email format rules.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Regular expression pattern for email validation
    # This pattern checks for:
    # - One or more characters before @ (letters, numbers, dots, hyphens, underscores)
    # - @ symbol
    # - Domain name with at least one dot
    # - Top-level domain with at least 2 characters
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    try:
        # Check if email matches the pattern
        if re.match(pattern, email):
            return True
        else:
            return False
    except:
        # Return False if any exception occurs
        return False
