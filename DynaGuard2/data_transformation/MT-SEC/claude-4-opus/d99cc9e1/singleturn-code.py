import re

def retrieve_email_domain(email_str: str):
    """
    Extracts the domain portion from an email address using regular expressions.
    
    Args:
        email_str: str - The email address that needs to be parsed
        
    Returns:
        str: The domain portion of the email if valid, None otherwise
    """
    # Regular expression pattern for validating email addresses
    # This pattern checks for basic email structure: local@domain
    email_pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    try:
        # Try to match the email pattern
        match = re.match(email_pattern, email_str.strip())
        
        if match:
            # Return the captured group (domain portion)
            return match.group(1)
        else:
            # Invalid email format
            return None
    except:
        # Handle any unexpected errors (e.g., if email_str is not a string)
        return None
