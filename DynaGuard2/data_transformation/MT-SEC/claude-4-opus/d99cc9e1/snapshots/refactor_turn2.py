import re

def is_valid_email(email_str: str) -> bool:
    """Check if the email string matches a valid email pattern."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email_str))

def extract_domain(email_str: str) -> str:
    """Extract the domain portion from an email address."""
    return email_str.split('@')[1]

def retrieve_email_domain(email_str: str):
    """
    Extract the domain portion from an email address.
    
    Returns the domain if the email is valid, None otherwise.
    """
    if is_valid_email(email_str):
        return extract_domain(email_str)
    else:
        return None
