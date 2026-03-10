import re

def is_valid_email(email):
    # Regular expression pattern for validating email addresses
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(pattern, email):
        return True
    else:
        return False

def get_email_username(email):
    # Regular expression pattern for validating email addresses
    pattern = r'^([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Try to match the email pattern
    match = re.match(pattern, email)
    
    if match:
        # Return the username part (first capturing group)
        return match.group(1)
    else:
        return None
