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
        ValueError: If redirect_url doesn't start with https://
    """
    # Validate that redirect_url uses HTTPS
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure URL: redirect_url must start with https://")
    
    # Check if the endpoint already has query parameters
    if '?' in auth_endpoint:
        # Append uid and redirect_url as additional parameters
        return f"{auth_endpoint}&uid={uid}&redirect_url={redirect_url}"
    else:
        # Add uid and redirect_url as the first query parameters
        return f"{auth_endpoint}?uid={uid}&redirect_url={redirect_url}"
