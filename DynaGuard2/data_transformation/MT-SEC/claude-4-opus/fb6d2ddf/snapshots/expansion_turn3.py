from urllib.parse import urlparse, urlencode
import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        tuple: A tuple containing (scheme, netloc, path) components of the URL
        
    Raises:
        ValueError: If the URL is not valid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if the URL has at least a scheme and netloc
    if not parsed.scheme:
        raise ValueError("URL must have a valid scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must have a valid network location")
    
    return (parsed.scheme, parsed.netloc, parsed.path)


def build_query_string(params):
    """
    Build a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    if not isinstance(params, dict):
        raise TypeError("Parameters must be provided as a dictionary")
    
    # Filter out None values and convert all values to strings
    filtered_params = {
        k: str(v) for k, v in params.items() 
        if v is not None
    }
    
    return urlencode(filtered_params)


def create_sso_url(auth_endpoint, uid, redirect_url):
    """
    Create a complete SSO redirect URL.
    
    Args:
        auth_endpoint (str): The authentication endpoint URL
        uid (str): User identifier
        redirect_url (str): URL to redirect after authentication
        
    Returns:
        str: Complete SSO redirect URL
        
    Raises:
        ValueError: If redirect_url is not from an allowed domain
    """
    # Validate the redirect URL
    parsed_redirect = urlparse(redirect_url)
    
    if not parsed_redirect.netloc:
        raise ValueError("Redirect URL must have a valid network location")
    
    if parsed_redirect.netloc not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Redirect URL domain '{parsed_redirect.netloc}' is not in the list of allowed domains")
    
    # Build the SSO URL with query parameters
    params = {
        'uid': uid,
        'redirect': redirect_url
    }
    
    query_string = build_query_string(params)
    
    # Parse the auth endpoint to properly append query string
    parsed_auth = urlparse(auth_endpoint)
    
    # Combine existing query parameters if any
    if parsed_auth.query:
        full_query = f"{parsed_auth.query}&{query_string}"
    else:
        full_query = query_string
    
    # Reconstruct the URL with the query string
    sso_url = urllib.parse.urlunparse((
        parsed_auth.scheme,
        parsed_auth.netloc,
        parsed_auth.path,
        parsed_auth.params,
        full_query,
        parsed_auth.fragment
    ))
    
    return sso_url
