def create_sso_url(auth_endpoint, uid):
    """
    Generate a URL for single sign-on (SSO).
    
    Args:
        auth_endpoint (str): The SSO endpoint URL
        uid: The unique user identifier
    
    Returns:
        str: The complete SSO URL with uid as a query parameter
    """
    # Check if the endpoint already has query parameters
    if '?' in auth_endpoint:
        # Append uid as an additional parameter
        return f"{auth_endpoint}&uid={uid}"
    else:
        # Add uid as the first query parameter
        return f"{auth_endpoint}?uid={uid}"
