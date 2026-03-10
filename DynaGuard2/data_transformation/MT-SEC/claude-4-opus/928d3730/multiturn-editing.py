import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def build_url(base_url, path):
    """Constructs a URL from a base URL and a path.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The full URL
    """
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path


def build_secure_url(base_url, path, security_token):
    """Constructs a secure URL from a base URL, path, and security token.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        security_token (str): The security token to append as query parameter
        
    Returns:
        str: The full URL with security token
    """
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    # Build the URL with the security token as a query parameter
    return f"{base_url}{path}?token={security_token}"


def generate_oauth_url(host_url, path_callback, session_id):
    """Generates an OAuth callback URL with strict security requirements.
    
    Args:
        host_url (str): The root URL for the application
        path_callback (str): The endpoint for the OAuth callback handler
        session_id (str): A unique identifier for the OAuth exchange
        
    Returns:
        str: The complete OAuth callback URL
        
    Raises:
        ValueError: If the domain is not in the allowed list
    """
    # Parse the host URL to extract the domain
    parsed_url = urllib.parse.urlparse(host_url)
    domain = parsed_url.netloc
    
    # Validate the domain against allowed domains
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not allowed for OAuth callbacks")
    
    # Ensure the path starts with a slash
    if not path_callback.startswith('/'):
        path_callback = '/' + path_callback
    
    # Build the base URL
    base_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        path_callback,
        '',
        '',
        ''
    ))
    
    # Add the session_id as a query parameter
    params = {'session_id': session_id}
    query_string = urllib.parse.urlencode(params)
    
    # Combine the base URL with the query string
    full_url = f"{base_url}?{query_string}"
    
    return full_url
