import re


def get_email_domain(addr: str):
    """
    Extracts the domain portion of a given email address using regular expressions.

    Args:
        addr: str, the email address to examine.

    Returns:
        If the email address is correctly structured, returns its domain part as a string.
        If not, returns None.
    """
    # Regex to validate the email structure and capture the domain.
    # The pattern is composed of the following parts:
    #   - `^[a-zA-Z0-9._%+-]+` : Matches the local part (username).
    #   - `@`                   : Literal '@' symbol.
    #   - `([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})` : Capturing group for the domain.
    #     - `[a-zA-Z0-9.-]+`  : Domain name (e.g., "example").
    #     - `\.`              : Literal '.' symbol.
    #     - `[a-zA-Z]{2,}`    : Top-level domain (e.g., "com").
    #   - `$`                   : Matches the end of the string.
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"  # Defines the regex pattern for a valid email structure and captures the domain.
    
    match = re.match(pattern, addr)  # Attempts to match the regex pattern from the beginning of the addr string.
    
    # Guard clause for invalid email format
    if not match:
        return None  # If the pattern does not match, the email is invalid; return None.
        
    # If a match is found, match.group(1) will contain the captured domain part of the email.
    return match.group(1)  # Return the extracted domain.
