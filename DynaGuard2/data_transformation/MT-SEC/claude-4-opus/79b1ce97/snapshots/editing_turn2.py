from urllib.parse import urlparse, urlencode, urlunparse

# Predefined set of approved domains
APPROVED_DOMAINS = {
    'example.com',
    'trusted-site.org',
    'safe-domain.net',
    'approved-host.io'
}

# Predefined set of approved endpoints
APPROVED_ENDPOINTS = {
    '/api/users',
    '/api/products',
    '/auth/login',
    '/auth/logout',
    '/dashboard',
    '/profile'
}

def validate_url_domain(url, endpoint, query_params=None):
    """
    Validates if a URL's domain and endpoint are in the approved lists.
    
    Args:
        url (str): The URL to validate
        endpoint (str): The endpoint to validate
        query_params (dict, optional): Query parameters to append to the URL
        
    Raises:
        ValueError: If the domain or endpoint is not in the approved lists
        
    Returns:
        str: The validated URL with endpoint and query parameters
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        if domain not in APPROVED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not in the approved domains list")
            
        # Ensure endpoint starts with /
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
            
        if endpoint not in APPROVED_ENDPOINTS:
            raise ValueError(f"Endpoint '{endpoint}' is not in the approved endpoints list")
        
        # Build the query string
        query_string = ''
        if query_params:
            query_string = urlencode(query_params)
        
        # Construct the validated URL
        validated_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            endpoint,
            '',
            query_string,
            ''
        ))
        
        return validated_url
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Invalid URL format: {url}")
