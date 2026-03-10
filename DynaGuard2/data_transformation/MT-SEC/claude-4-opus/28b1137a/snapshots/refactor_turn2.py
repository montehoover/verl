import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def validate_redirect_domain(redirect_url):
    """
    Validates that the redirect URL's domain is in the allowed list.
    
    Args:
        redirect_url: The URL to validate
        
    Returns:
        The domain if valid
        
    Raises:
        ValueError: If domain is not in the allowed list
    """
    parsed_url = urllib.parse.urlparse(redirect_url)
    domain = parsed_url.netloc
    
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    return domain

def construct_sso_url(base_path, query_params):
    """
    Constructs a URL by appending query parameters to a base path.
    
    Args:
        base_path: The base SSO path
        query_params: Dictionary of query parameters
        
    Returns:
        The complete URL with query parameters
    """
    query_string = urllib.parse.urlencode(query_params)
    
    if '?' in base_path:
        return f"{base_path}&{query_string}"
    else:
        return f"{base_path}?{query_string}"

def build_sso_url(sso_path, uid, redirect_url):
    # Validate the redirect URL domain
    validate_redirect_domain(redirect_url)
    
    # Build the query parameters
    params = {
        'uid': uid,
        'redirect': redirect_url
    }
    
    # Construct and return the SSO URL
    return construct_sso_url(sso_path, params)
