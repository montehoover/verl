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
    # Regular expression for validating email addresses
    # This pattern ensures a properly formatted email with:
    # - A local part containing letters, numbers, and common email characters
    # - An @ symbol separating local and domain parts
    # - A domain with at least one subdomain and a valid TLD
    email_validation_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Pattern components explained:
    # ^[a-zA-Z0-9._%+-]+  : Start of string, followed by one or more valid local part chars
    #                       (letters, numbers, dots, underscores, percent, plus, hyphen)
    # @                   : Required separator between local and domain parts
    # [a-zA-Z0-9.-]+      : Domain name - one or more letters, numbers, dots, or hyphens
    # \.                  : Literal dot before the top-level domain
    # [a-zA-Z]{2,}$       : TLD must be 2+ letters, anchored to end of string
    
    # Early return for invalid email format
    if not re.match(email_validation_pattern, email):
        return None
    
    # Extract and return domain part for valid emails
    # Using split() is safe here since we've already validated the @ symbol exists
    domain = email.split('@')[1]
    return domain
