from urllib.parse import urlparse

# List of permitted domains
PERMITTED_DOMAINS = [
    'example.com',
    'secure-site.com',
    'trusted-domain.org',
    'api.example.com',
    'www.example.com'
]

def validate_url(url):
    """
    Validates a URL and returns True if valid, raises ValueError if invalid.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    try:
        result = urlparse(url)
        
        # Check if scheme and netloc are present (minimum requirements for a valid URL)
        if not result.scheme:
            raise ValueError("URL must have a scheme (e.g., http, https)")
        
        if not result.netloc:
            raise ValueError("URL must have a network location (domain)")
        
        # Check for valid schemes
        valid_schemes = ['http', 'https', 'ftp', 'ftps']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(valid_schemes)}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def is_domain_permitted(url):
    """
    Checks if a URL's domain is in the list of permitted domains.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the domain is permitted, False otherwise
    """
    try:
        # First validate the URL
        validate_url(url)
        
        # Parse the URL to extract the domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove port number if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check if domain is in permitted list
        return domain in [d.lower() for d in PERMITTED_DOMAINS]
        
    except ValueError:
        # If URL is invalid, domain is not permitted
        return False
