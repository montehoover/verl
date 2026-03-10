import re

def get_email_domain(mail_address: str):
    """
    Extracts the domain portion from an email address using regular expressions.
    
    Args:
        mail_address: str - The email address that needs to be parsed
        
    Returns:
        str: The domain portion if the email is valid, None otherwise
    """
    # Email regex pattern that validates email structure
    # This pattern checks for:
    # - Local part: alphanumeric, dots, hyphens, underscores
    # - @ symbol
    # - Domain part: alphanumeric, dots, hyphens (with proper structure)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    try:
        match = re.match(email_pattern, mail_address)
        if match:
            return match.group(1)
        else:
            return None
    except:
        # Handle any unexpected errors (though regex shouldn't raise exceptions for strings)
        return None
