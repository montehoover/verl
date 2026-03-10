import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generates a Single Sign-On (SSO) redirect URL for a third-party service integration.

    Combines the SSO endpoint with user-specific parameters and a return URL.

    Args:
        sso_endpoint: The SSO service endpoint.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL.

    Raises:
        ValueError: If the return_url is not from an allowed domain.
    """
    parsed_return_url = urllib.parse.urlparse(return_url)
    if parsed_return_url.hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(
            f"Return URL domain '{parsed_return_url.hostname}' is not allowed. "
            f"Allowed domains are: {', '.join(ALLOWED_RETURN_DOMAINS)}"
        )

    params = {
        'user_id': user_id,
        'return_to': return_url
    }
    query_string = urllib.parse.urlencode(params)
    
    # Ensure the sso_endpoint and query_string are combined correctly,
    # handling cases where sso_endpoint might or might not have existing query params.
    parsed_sso_endpoint = urllib.parse.urlparse(sso_endpoint)
    if parsed_sso_endpoint.query:
        separator = '&'
    else:
        separator = '?'
        
    redirect_url = f"{sso_endpoint}{separator}{query_string}"
    
    return redirect_url
