import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _validate_redirect_url(redirect_url: str, allowed_domains: set) -> None:
    """
    Validates the redirect URL against a set of allowed domains.

    Args:
        redirect_url: The URL to validate.
        allowed_domains: A set of allowed hostnames.

    Raises:
        ValueError: If the redirect_url's domain is not in allowed_domains.
    """
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.hostname not in allowed_domains:
        raise ValueError(
            f"Redirect URL domain '{parsed_redirect_url.hostname}' is not allowed."
        )


def _construct_sso_url_with_params(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Constructs the full SSO URL with user ID and redirect URL as query parameters.

    Args:
        sso_path: The base SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL.
    """
    query_params = urllib.parse.urlencode({'user_id': uid, 'return_to': redirect_url})
    
    if '?' not in sso_path:
        sso_url = f"{sso_path}?{query_params}"
    else:
        sso_url = f"{sso_path}&{query_params}"
        
    return sso_url


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
    _validate_redirect_url(redirect_url, ALLOWED_RETURN_DOMAINS)
    return _construct_sso_url_with_params(sso_path, uid, redirect_url)
