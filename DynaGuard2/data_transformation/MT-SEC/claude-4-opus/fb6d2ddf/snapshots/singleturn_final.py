import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint, uid, redirect_url):
    """
    Constructs an SSO redirect URL that facilitates authentication with an external identity provider.
    
    Args:
        auth_endpoint: a string indicating the SSO endpoint URL
        uid: a string representing the unique user ID
        redirect_url: a string pointing to the return URL after SSO completes
    
    Returns:
        A string containing the full SSO redirect URL
    
    Raises:
        ValueError: raised if the redirect_url's domain is not part of the approved domains list
    """
    # Parse the redirect URL to extract the domain
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    redirect_domain = parsed_redirect.netloc
    
    # Validate that the redirect domain is in the allowed list
    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Redirect domain '{redirect_domain}' is not in the allowed domains list")
    
    # Construct the query parameters
    params = {
        'user': uid,
        'return': redirect_url
    }
    
    # Build the complete SSO URL
    sso_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
    
    return sso_url
