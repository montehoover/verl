from urllib.parse import urlparse, urlencode, quote_plus
import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

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


def construct_oauth_callback_uri(root_url, callback_uri, session_state):
    """
    Constructs a complete OAuth callback URL with domain validation.
    
    Args:
        root_url (str): The base URL
        callback_uri (str): The callback URI path
        session_state (str): The session state parameter
        
    Returns:
        str: Complete OAuth callback URL
        
    Raises:
        ValueError: If the callback URL's domain is not in the allowed list
    """
    # Combine root_url and callback_uri
    if callback_uri.startswith('http://') or callback_uri.startswith('https://'):
        full_url = callback_uri
    else:
        full_url = urllib.parse.urljoin(root_url, callback_uri)
    
    # Parse the URL to extract domain
    parsed = urllib.parse.urlparse(full_url)
    domain = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {domain} is not in the allowed callback domains")
    
    # Add session_state as query parameter
    params = {'state': session_state}
    
    # Parse existing query parameters
    existing_params = urllib.parse.parse_qs(parsed.query)
    
    # Merge with new parameters
    for key, values in existing_params.items():
        if key not in params:
            params[key] = values[0] if len(values) == 1 else values
    
    # Construct new query string
    new_query = urllib.parse.urlencode(params)
    
    # Rebuild URL with new query parameters
    new_parsed = parsed._replace(query=new_query)
    
    return urllib.parse.urlunparse(new_parsed)
