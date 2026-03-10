import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint, uid, redirect_url):
    """
    Generate a URL for single sign-on (SSO).
    
    Args:
        auth_endpoint (str): The SSO endpoint URL
        uid: The unique user identifier
        redirect_url (str): The URL to redirect to after authentication
    
    Returns:
        str: The complete SSO URL with uid and redirect_url as query parameters
    
    Raises:
        ValueError: If redirect_url doesn't start with https:// or if domain is not allowed
    """
    # Validate that redirect_url uses HTTPS
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure URL: redirect_url must start with https://")
    
    # Parse the redirect_url to extract the domain
    parsed_url = urllib.parse.urlparse(redirect_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain not allowed: {domain}")
    
    # Check if the endpoint already has query parameters
    if '?' in auth_endpoint:
        # Append uid and redirect_url as additional parameters
        return f"{auth_endpoint}&uid={uid}&redirect_url={redirect_url}"
    else:
        # Add uid and redirect_url as the first query parameters
        return f"{auth_endpoint}?uid={uid}&redirect_url={redirect_url}"
