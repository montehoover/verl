import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Generates an OAuth callback URL with domain validation and session ID.

    Args:
        host_url: The root URL for the application (e.g., "https://auth.myapp.com").
        path_callback: The endpoint for the OAuth callback handler (e.g., "/callback").
        session_id: A unique identifier for the OAuth exchange.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the domain of host_url is not in ALLOWED_CALLBACK_DOMAINS.
    """
    parsed_host_url = urllib.parse.urlparse(host_url)
    if parsed_host_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Domain {parsed_host_url.hostname} is not an allowed callback domain."
        )

    # Ensure path_callback starts with a slash if it doesn't have one
    if not path_callback.startswith('/'):
        path_callback = '/' + path_callback
    
    # Construct the full path
    full_path = parsed_host_url.path.rstrip('/') + path_callback

    # Build query parameters
    query_params = urllib.parse.urlencode({'session_id': session_id})

    # Construct the final URL
    # scheme, netloc, path, params, query, fragment
    final_url = urllib.parse.urlunparse(
        (parsed_host_url.scheme, parsed_host_url.netloc, full_path, '', query_params, '')
    )
    return final_url
