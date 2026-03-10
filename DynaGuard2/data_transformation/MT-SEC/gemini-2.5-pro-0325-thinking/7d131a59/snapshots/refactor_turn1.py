import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Generates an OAuth callback URL for a third-party authentication process.

    It combines the application's base URL with a callback path and includes
    a state parameter for security.

    Args:
        base_url: The base URL of the application.
        callback_path: The path for the OAuth callback.
        state: A unique state parameter for the OAuth flow.

    Returns:
        The complete OAuth callback URL.

    Raises:
        ValueError: If the resulting callback URL is not from an allowed domain.
    """
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_base_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_base_url.hostname}' is not an allowed callback domain."
        )

    # Ensure callback_path starts with a slash if base_url doesn't end with one,
    # and vice-versa to prevent double slashes or missing slashes.
    if base_url.endswith('/') and callback_path.startswith('/'):
        effective_callback_path = callback_path[1:]
    elif not base_url.endswith('/') and not callback_path.startswith('/'):
        effective_callback_path = '/' + callback_path
    else:
        effective_callback_path = callback_path
    
    full_callback_url_without_state = urllib.parse.urljoin(base_url, effective_callback_path)

    # Add the state parameter
    url_parts = list(urllib.parse.urlparse(full_callback_url_without_state))
    query = dict(urllib.parse.parse_qsl(url_parts[4]))
    query['state'] = state
    url_parts[4] = urllib.parse.urlencode(query)

    return urllib.parse.urlunparse(url_parts)
