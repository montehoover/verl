import re


def fetch_email_domain(mail_id: str):
    """
    Extract the domain portion from an email address using regular expressions.
    
    Args:
        mail_id (str): The email address that needs to be parsed.
        
    Returns:
        str or None: If the input email address is valid, return the domain 
                     portion as a string. Otherwise, return None.
    """
    # Regex pattern breakdown:
    # ^[a-zA-Z0-9._%+-]+  - Matches the local part (before @): alphanumeric, dots, underscores, percent, plus, hyphen
    # @                   - Matches the @ symbol literally
    # ([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$ - Captures the domain part:
    #   [a-zA-Z0-9.-]+    - Domain name: alphanumeric, dots, hyphens
    #   \.                - Literal dot before TLD
    #   [a-zA-Z]{2,}      - Top-level domain: at least 2 letters
    #   $                 - End of string anchor
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    match = re.match(pattern, mail_id)
    
    # Early return with domain if match found
    if not match:
        return None
    
    return match.group(1)
