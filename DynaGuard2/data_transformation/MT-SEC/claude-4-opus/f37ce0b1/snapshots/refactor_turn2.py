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
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    match = re.match(pattern, mail_id)
    
    if match:
        return match.group(1)
    
    return None
