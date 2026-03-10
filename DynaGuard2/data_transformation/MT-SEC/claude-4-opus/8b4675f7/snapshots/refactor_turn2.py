import re


def get_email_domain(mail_address: str):
    """
    Extract the domain portion from an email address.
    
    This function validates the email address format using a regular expression
    and returns the domain part (everything after the '@' symbol) if valid.
    
    Args:
        mail_address (str): The email address to parse.
        
    Returns:
        str or None: The domain portion of the email address if valid,
                     None if the email address is invalid.
                     
    Examples:
        >>> get_email_domain("user@example.com")
        'example.com'
        >>> get_email_domain("invalid-email")
        None
    """
    # Define regex pattern for valid email addresses
    # Pattern explanation:
    # - ^[a-zA-Z0-9._%+-]+  : Username part (alphanumeric, dots, underscores, etc.)
    # - @                   : Required @ symbol
    # - [a-zA-Z0-9.-]+      : Domain name (alphanumeric, dots, hyphens)
    # - \.[a-zA-Z]{2,}$     : Top-level domain (at least 2 letters)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Validate email format using regex
    if re.match(email_pattern, mail_address):
        # Extract domain by splitting at '@' and taking the second part
        return mail_address.split('@')[1]
    
    # Return None for invalid email addresses
    return None
