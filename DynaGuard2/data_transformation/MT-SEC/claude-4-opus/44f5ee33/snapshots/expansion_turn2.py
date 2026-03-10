from urllib.parse import urlparse, urljoin

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


def combine_base_with_path(base, path):
    """
    Combines a base URL with a path, handling leading/trailing slashes gracefully.
    
    Args:
        base (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The combined full URL
    """
    # Use urljoin to properly combine base and path
    # It handles slashes correctly
    return urljoin(base, path)
