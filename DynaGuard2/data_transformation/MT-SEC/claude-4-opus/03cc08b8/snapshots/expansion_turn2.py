from urllib.parse import urlparse, urljoin, urlencode

def validate_url_domain(url):
    """
    Validate if a URL's domain is in the allowed list.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if domain is allowed, False otherwise
        
    Raises:
        ValueError: If the URL is invalid
    """
    # Predefined list of allowed domains
    ALLOWED_DOMAINS = [
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'localhost'
    ]
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if the URL has a valid scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        # Extract the domain (netloc may include port)
        domain = parsed.netloc.split(':')[0].lower()
        
        # Check if domain is in allowed list
        return domain in ALLOWED_DOMAINS
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {url}") from e


def build_url_with_params(base_url, path, params):
    """
    Build a complete URL with path and parameters.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append to the base URL
        params (dict): Dictionary of parameters to add as query string
        
    Returns:
        str: Complete URL with path and parameters
    """
    # Join base URL with path
    url = urljoin(base_url, path)
    
    # Add parameters if provided
    if params:
        # Encode parameters
        query_string = urlencode(params)
        # Append to URL
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}{query_string}"
    
    return url
