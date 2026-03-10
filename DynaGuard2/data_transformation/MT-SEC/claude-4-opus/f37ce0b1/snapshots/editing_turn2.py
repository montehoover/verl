import re

def fetch_email_domain(url):
    """
    Extract the full subdomain and domain from a URL.
    
    Args:
        url (str): The URL to extract from
        
    Returns:
        str: The subdomain and domain (e.g., "sub.example.com") or None if invalid
    """
    # Regular expression to match the domain part of a URL
    pattern = r'https?://([^/]+)'
    
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return None
