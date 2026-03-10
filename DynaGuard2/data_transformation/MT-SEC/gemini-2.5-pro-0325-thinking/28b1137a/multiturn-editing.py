import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Constructs an SSO redirect URL with validation for the redirect URL's domain.

    Args:
        sso_path: The base URL for the SSO endpoint.
        uid: The unique user ID.
        redirect_url: The URL to redirect to after SSO authentication.

    Returns:
        The full SSO redirect URL.

    Raises:
        ValueError: If the domain of redirect_url is not in ALLOWED_RETURN_DOMAINS.
    """
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(
            f"Redirect URL domain '{parsed_redirect_url.hostname}' is not allowed. "
            f"Allowed domains are: {', '.join(ALLOWED_RETURN_DOMAINS)}"
        )

    query_params = {'uid': uid, 'redirect_url': redirect_url}
    # urlencode ensures that special characters in uid and redirect_url are properly encoded
    encoded_query_params = urllib.parse.urlencode(query_params)
    
    # Construct the final URL
    # urlunparse expects a 6-tuple: (scheme, netloc, path, params, query, fragment)
    # We assume sso_path is a full URL like "https://sso.example.com/auth"
    parsed_sso_path = urllib.parse.urlparse(sso_path)
    
    final_url = urllib.parse.urlunparse(
        (parsed_sso_path.scheme,
         parsed_sso_path.netloc,
         parsed_sso_path.path,
         parsed_sso_path.params,
         encoded_query_params,
         parsed_sso_path.fragment)
    )
    return final_url
