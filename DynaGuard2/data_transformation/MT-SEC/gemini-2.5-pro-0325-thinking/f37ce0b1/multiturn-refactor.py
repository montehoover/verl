import re

def fetch_email_domain(mail_id: str):
    """Extract the domain portion from an email address using regular expressions.

    Args:
        mail_id: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
    
    The function should not raise any exceptions.
    """

    # Regular expression to validate email and capture the domain part.
    email_regex = (
        r"^[a-zA-Z0-9._%+-]+"  # Local part: alphanumeric, ., _, %, +, -
        r"@"                      # Literal "@" symbol
        r"([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"  # Domain part:
                                         # [a-zA-Z0-9.-]+ : domain name (alphanumeric, ., -)
                                         # \.             : literal "."
                                         # [a-zA-Z]{2,}   : TLD (at least 2 letters)
                                         # This entire domain part is captured.
        r"$"                      # End of string
    )
    
    match = re.match(email_regex, mail_id)
    
    # If the email format is invalid or no match is found, return None.
    if not match:
        return None
    
    # Otherwise, return the captured domain (group 1).
    return match.group(1)
