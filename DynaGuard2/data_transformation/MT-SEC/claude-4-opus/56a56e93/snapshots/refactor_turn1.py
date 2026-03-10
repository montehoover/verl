import re

def fetch_email_domain(address: str):
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    match = re.match(pattern, address)
    if match:
        return match.group(1)
    return None
