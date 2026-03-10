from urllib.parse import urlparse

def validate_url_domain(url):
    """
    Validates if a URL's domain is in the list of allowed domains.
    
    Args:
        url (str): The URL to validate
        
    Raises:
        ValueError: If the domain is not in the allowed list
    """
    # List of allowed domains
    allowed_domains = [
        'example.com',
        'api.example.com',
        'staging.example.com',
        'localhost',
        '127.0.0.1'
    ]
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain (netloc)
    domain = parsed_url.netloc.lower()
    
    # Remove port number if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if domain is in allowed list
    if domain not in allowed_domains:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    return True
