import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_sso_redirect(sso_endpoint, user_id, return_url):
    """
    Generates a Single Sign-On (SSO) redirect URL for a third-party service integration.
    
    Args:
        sso_endpoint: a string representing the SSO service endpoint.
        user_id: a string representing the user's unique identifier.
        return_url: a string representing the URL to return to after SSO.
    
    Returns:
        A string that is the complete SSO redirect URL.
    
    Raises:
        ValueError: if the return_url is not from an allowed domain.
    """
    # Parse the return URL to extract the domain
    parsed_url = urllib.parse.urlparse(return_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Return URL domain '{domain}' is not allowed")
    
    # Construct the query parameters
    params = {
        'user': user_id,
        'return': return_url
    }
    
    # Encode the parameters and append to the SSO endpoint
    query_string = urllib.parse.urlencode(params)
    redirect_url = f"{sso_endpoint}?{query_string}"
    
    return redirect_url
