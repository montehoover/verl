import re

def retrieve_email_domain(email_str: str):
    # Regular expression pattern for validating email addresses
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(email_pattern, email_str):
        # Extract the domain portion (everything after '@')
        domain = email_str.split('@')[1]
        return domain
    else:
        return None
