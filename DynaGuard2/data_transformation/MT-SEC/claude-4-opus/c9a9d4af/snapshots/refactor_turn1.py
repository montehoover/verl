import re

def retrieve_domain_from_email(email_address: str):
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    match = re.match(pattern, email_address)
    if match:
        return match.group(1)
    return None
