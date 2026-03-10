import re


def extract_email_domain(email: str) -> str:
    """
    Extract the domain part from an email address.
    
    This function validates an email address using a regular expression pattern
    and extracts the domain portion (everything after the '@' symbol) if the
    email is valid.
    
    Args:
        email (str): The email address to process.
        
    Returns:
        str or None: The domain part of the email if valid, None otherwise.
        
    Examples:
        >>> extract_email_domain("user@example.com")
        'example.com'
        >>> extract_email_domain("invalid.email")
        None
    """
    # Define regex pattern for email validation
    # Pattern breakdown:
    # - ^[a-zA-Z0-9._%+-]+ : Local part (before @) - alphanumeric and special chars
    # - @ : Required @ symbol
    # - [a-zA-Z0-9.-]+ : Domain name - alphanumeric, dots, and hyphens
    # - \.[a-zA-Z]{2,}$ : Top-level domain - dot followed by 2+ letters
    email_validation_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Validate email format using regex
    if re.match(email_validation_pattern, email):
        # Split email at '@' symbol and return domain part (index 1)
        domain = email.split('@')[1]
        return domain
    else:
        # Return None for invalid email addresses
        return None
