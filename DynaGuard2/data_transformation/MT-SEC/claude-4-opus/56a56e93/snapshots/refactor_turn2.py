import re


def fetch_email_domain(address: str):
    """
    Extract the domain from a valid email address.
    
    Args:
        address (str): The email address to process and analyze.
        
    Returns:
        str or None: The domain portion of the email address (after the '@' sign)
                     if the email is valid, otherwise None.
    """
    # Regular expression pattern to validate email and capture domain
    # Pattern breakdown:
    # ^[a-zA-Z0-9._%+-]+ : Matches username part (alphanumeric, dots, underscores, etc.)
    # @ : Literal @ symbol
    # ([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$ : Captures domain part in group 1
    #   - [a-zA-Z0-9.-]+ : Domain name (alphanumeric, dots, hyphens)
    #   - \. : Literal dot
    #   - [a-zA-Z]{2,} : Top-level domain (at least 2 letters)
    #   - $ : End of string
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    # Attempt to match the pattern against the email address
    match = re.match(pattern, address)
    
    # Return the captured domain group if match is successful
    if match:
        return match.group(1)
    
    return None
