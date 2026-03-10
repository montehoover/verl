import re

def get_email_domain(addr: str):
    # Regular expression pattern for valid email address
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email address matches the pattern
    if re.match(email_pattern, addr):
        # Extract and return the domain part (everything after @)
        return addr.split('@')[1]
    else:
        return None
