import re

def retrieve_email_domain(email_str: str) -> str:
    # Regular expression pattern for validating email addresses
    email_pattern = r'^([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    match = re.match(email_pattern, email_str)
    if match:
        return match.group(1)
    else:
        return None
