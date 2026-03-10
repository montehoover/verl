import re

def extract_email_domain(email: str) -> str:
    # Regular expression pattern for validating email addresses
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(email_pattern, email):
        # Extract and return the domain part (everything after '@')
        return email.split('@')[1]
    else:
        # Return None if the email is not valid
        return None
