from urllib.parse import urlparse, urlencode, urljoin, urlunparse

def validate_url_domain(url):
    """
    Validates whether the domain of a URL is in the allowed list.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is allowed, False otherwise
        
    Raises:
        ValueError: If the URL is invalid or malformed
    """
    # Predefined set of allowed domains
    ALLOWED_DOMAINS = {
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'internal.company.com'
    }
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if scheme and netloc are present (basic URL validation)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        # Extract the domain (netloc may include port, so we split)
        domain = parsed.netloc.split(':')[0].lower()
        
        # Remove 'www.' prefix if present for comparison
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is in allowed list
        return domain in ALLOWED_DOMAINS
        
    except Exception as e:
        raise ValueError(f"Error parsing URL '{url}': {str(e)}")


def build_url_with_params(base_url, path, params):
    """
    Constructs a complete URL with path and parameters.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append to the base URL
        params (dict): Dictionary of parameters to include in the URL
        
    Returns:
        str: Complete URL with path and encoded parameters
    """
    # Join base URL with path
    url_with_path = urljoin(base_url, path)
    
    # Parse the URL to modify it
    parsed = urlparse(url_with_path)
    
    # Encode the parameters
    query_string = urlencode(params)
    
    # Reconstruct the URL with the query string
    final_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        query_string,
        parsed.fragment
    ))
    
    return final_url
