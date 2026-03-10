import re

def fetch_email_domain(address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.
    
    Args:
        address: str - the email address to process and analyze
    
    Returns:
        str - the domain (portion after '@') if the email is valid
        None - if the email structure is not valid
    """
    # Regular expression pattern for validating email structure
    # This pattern checks for:
    # - One or more characters before @ (local part)
    # - Exactly one @ symbol
    # - One or more characters after @ (domain part)
    email_pattern = r'^[^@]+@([^@]+)$'
    
    match = re.match(email_pattern, address)
    
    if match:
        # Extract the domain (captured group 1)
        return match.group(1)
    else:
        # Invalid email structure
        return None
