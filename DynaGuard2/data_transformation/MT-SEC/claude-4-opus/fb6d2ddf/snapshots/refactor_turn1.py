import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint, uid, redirect_url):
    """
    Constructs an SSO redirect URL that facilitates authentication with an external identity provider.
    
    Args:
        auth_endpoint: A string indicating the SSO endpoint URL.
        uid: A string representing the unique user ID.
        redirect_url: A string pointing to the return URL after SSO completes.
    
    Returns:
        A string containing the full SSO redirect URL.
    
    Raises:
        ValueError: Raised if the redirect_url's domain is not part of the approved domains list.
    """
    # Parse the redirect URL to extract the domain
    parsed_url = urllib.parse.urlparse(redirect_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    # Construct the SSO URL with query parameters
    params = {
        'uid': uid,
        'redirect': redirect_url
    }
    
    # Parse the auth endpoint and add query parameters
    parsed_endpoint = urllib.parse.urlparse(auth_endpoint)
    query_params = urllib.parse.parse_qs(parsed_endpoint.query)
    query_params.update(params)
    
    # Build the final URL
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    sso_url = urllib.parse.urlunparse((
        parsed_endpoint.scheme,
        parsed_endpoint.netloc,
        parsed_endpoint.path,
        parsed_endpoint.params,
        new_query,
        parsed_endpoint.fragment
    ))
    
    return sso_url
