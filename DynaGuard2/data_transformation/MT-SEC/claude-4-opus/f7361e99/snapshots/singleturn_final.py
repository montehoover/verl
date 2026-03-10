import re

def extract_email_domain(email: str) -> str:
    """
    Extract the domain part from an email address using a regular expression.
    
    Args:
        email: str - The input email address to be processed
        
    Returns:
        str or None - The domain part if valid email, None otherwise
    """
    # Regular expression pattern for validating email addresses
    # This pattern checks for:
    # - One or more characters before @ (local part)
    # - @ symbol
    # - One or more characters after @ (domain part)
    email_pattern = r'^[^@]+@([^@]+)$'
    
    # Try to match the email pattern
    match = re.match(email_pattern, email)
    
    if match:
        # Extract the domain part (group 1)
        return match.group(1)
    else:
        # Return None if not a valid email
        return None
