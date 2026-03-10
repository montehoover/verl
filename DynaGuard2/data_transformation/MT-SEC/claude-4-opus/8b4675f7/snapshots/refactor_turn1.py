import re

def get_email_domain(mail_address: str):
    # Regular expression pattern for valid email addresses
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(email_pattern, mail_address):
        # Extract and return the domain part (after @)
        return mail_address.split('@')[1]
    else:
        return None
