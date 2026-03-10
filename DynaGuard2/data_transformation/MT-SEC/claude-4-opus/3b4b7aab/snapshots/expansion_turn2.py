from urllib.parse import urlparse, urlencode, quote_plus

def validate_url(url):
    """
    Validates a URL and checks if it belongs to a trusted domain.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid and from a trusted domain, False otherwise
        
    Raises:
        ValueError: If the URL is invalid
    """
    # Define trusted domains
    TRUSTED_DOMAINS = [
        'example.com',
        'trusted-site.org',
        'mycompany.com',
        'localhost'
    ]
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if URL has a valid scheme
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
        
        # Check if URL has a hostname
        if not parsed.netloc:
            raise ValueError("URL must have a valid hostname")
        
        # Extract the domain (remove port if present)
        domain = parsed.netloc.split(':')[0].lower()
        
        # Check if domain is in trusted list
        for trusted_domain in TRUSTED_DOMAINS:
            if domain == trusted_domain or domain.endswith('.' + trusted_domain):
                return True
        
        return False
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def construct_query_parameters(params):
    """
    Constructs a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of query parameters
        
    Returns:
        str: URL-encoded query string
    """
    return urlencode(params, safe='', quote_via=quote_plus)
