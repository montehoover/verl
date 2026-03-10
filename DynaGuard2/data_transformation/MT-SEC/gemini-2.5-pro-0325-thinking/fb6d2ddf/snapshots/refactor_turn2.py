import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _validate_redirect_domain(redirect_url: str, allowed_domains: set) -> None:
    """
    Validates if the redirect URL's domain is in the allowed list.

    Args:
        redirect_url: The URL to validate.
        allowed_domains: A set of allowed domain hostnames.

    Raises:
        ValueError: If the redirect_url's domain is not in allowed_domains.
    """
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.hostname not in allowed_domains:
        raise ValueError(
            f"Redirect URL domain '{parsed_redirect_url.hostname}' is not allowed."
        )


def _construct_sso_url_with_params(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Constructs the SSO URL with user ID and redirect URL as query parameters.

    Args:
        auth_endpoint: The base SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL string.
    """
    params = {
        'user_id': uid,
        'return_to': redirect_url
    }
    
    # Determine the correct separator based on whether auth_endpoint already has query params
    separator = '&' if '?' in auth_endpoint else '?'
    
    # Append query parameters to the auth_endpoint
    query_string = urllib.parse.urlencode(params)
    sso_url = f"{auth_endpoint}{separator}{query_string}"
    
    return sso_url


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
        ValueError: If the redirect_url's domain is not part of the
                    approved domains list.
    """
    _validate_redirect_domain(redirect_url, ALLOWED_RETURN_DOMAINS)
    
    sso_url = _construct_sso_url_with_params(auth_endpoint, uid, redirect_url)
    
    return sso_url
