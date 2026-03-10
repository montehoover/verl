import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Constructs an SSO redirect URL.

    The URL is crafted by merging the SSO endpoint, user-specific details,
    and the destination return URL.

    Args:
        auth_endpoint: The SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL.

    Raises:
        ValueError: If the redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(
            f"Redirect URL domain '{parsed_redirect_url.hostname}' is not allowed."
        )

    query_params = {
        'user': uid,
        'return': redirect_url
    }
    encoded_query_params = urllib.parse.urlencode(query_params)
    
    sso_url = f"{auth_endpoint}?{encoded_query_params}"
    return sso_url
