import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Constructs an SSO redirect URL.

    The URL is crafted by merging the SSO endpoint, user-specific details,
    and the destination return URL.

    Args:
        sso_path: The SSO endpoint URL.
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

    query_params = urllib.parse.urlencode({'user_id': uid, 'return_to': redirect_url})
    
    # Ensure sso_path ends with a '?' or '&' appropriately
    # For simplicity, we'll assume sso_path is a base path and we append '?'
    # A more robust solution might check if sso_path already has query params.
    if '?' not in sso_path:
        sso_url = f"{sso_path}?{query_params}"
    else:
        sso_url = f"{sso_path}&{query_params}"
        
    return sso_url
