import re


def is_valid_email(email_str: str) -> bool:
    """
    Check if the email string matches a valid email pattern.
    
    Args:
        email_str: The email address string to validate.
        
    Returns:
        bool: True if the email matches the valid pattern, False otherwise.
    """
    # Define regex pattern for valid email addresses
    # Pattern breakdown:
    # - ^[a-zA-Z0-9._%+-]+ : Local part (before @) with allowed characters
    # - @ : Required @ symbol
    # - [a-zA-Z0-9.-]+ : Domain name with allowed characters
    # - \.[a-zA-Z]{2,}$ : Top-level domain with at least 2 letters
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Use re.match to check if the entire string matches the pattern
    return bool(re.match(email_pattern, email_str))


def extract_domain(email_str: str) -> str:
    """
    Extract the domain portion from an email address.
    
    Args:
        email_str: The email address string containing an @ symbol.
        
    Returns:
        str: The domain portion of the email (everything after @).
    """
    # Split the email at @ symbol and return the second part (domain)
    return email_str.split('@')[1]


def retrieve_email_domain(email_str: str) -> str | None:
    """
    Extract the domain portion from an email address using regular expressions.
    
    This function validates the email format and extracts the domain portion
    (the part after the '@' symbol) for well-formed emails.
    
    Args:
        email_str: The email address that needs to be parsed.
        
    Returns:
        str | None: The domain portion as a string if the email is valid,
                    None if the email format is invalid.
    """
    # Use guard clause for early return on invalid email
    if not is_valid_email(email_str):
        return None
    
    # Extract and return the domain portion for valid emails
    domain = extract_domain(email_str)
    return domain
