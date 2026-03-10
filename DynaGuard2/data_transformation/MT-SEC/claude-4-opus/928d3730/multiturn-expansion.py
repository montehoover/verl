from urllib.parse import urlparse, urlencode, urljoin, urlunparse
import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

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


def generate_oauth_url(host_url, path_callback, session_id):
    """
    Generates a secure OAuth callback URL.
    
    Args:
        host_url (str): The host URL
        path_callback (str): The callback path
        session_id (str): The session ID to include in the URL
        
    Returns:
        str: Complete OAuth callback URL
        
    Raises:
        ValueError: If the callback domain is not in the allowed list
    """
    # Construct the callback URL
    callback_url = urllib.parse.urljoin(host_url, path_callback)
    
    # Parse the callback URL to extract the domain
    parsed = urllib.parse.urlparse(callback_url)
    domain = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Remove 'www.' prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check if domain is allowed
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed callback domains")
    
    # Add session_id as a query parameter
    params = {'session_id': session_id}
    query_string = urllib.parse.urlencode(params)
    
    # Reconstruct the URL with the session_id parameter
    final_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        query_string,
        parsed.fragment
    ))
    
    return final_url
