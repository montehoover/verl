import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Constructs an OAuth callback URL used during third-party authentication.

    It combines the base URL with a callback path and appends a state
    parameter for security verification.

    Args:
        root_url: The application's main base URL.
        callback_uri: The specific path for the OAuth callback endpoint.
        session_state: A unique value for ensuring the integrity of the OAuth request flow.

    Returns:
        A string representing the finalized OAuth callback URL.

    Raises:
        ValueError: If the computed callback URL's domain is not within
                    the list of allowed domains.
    """
    # Ensure callback_uri does not start with a slash if root_url ends with one,
    # or vice-versa, to prevent double slashes.
    if root_url.endswith('/') and callback_uri.startswith('/'):
        base_callback_url = root_url + callback_uri[1:]
    elif not root_url.endswith('/') and not callback_uri.startswith('/'):
        base_callback_url = root_url + '/' + callback_uri
    else:
        base_callback_url = root_url + callback_uri

    parsed_url = urllib.parse.urlparse(base_callback_url)
    domain = parsed_url.netloc

    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"The domain '{domain}' is not an allowed callback domain. "
            f"Allowed domains are: {', '.join(ALLOWED_CALLBACK_DOMAINS)}"
        )

    # Add the state parameter
    query_params = urllib.parse.urlencode({'state': session_state})
    final_url = urllib.parse.urlunparse(parsed_url._replace(query=query_params))

    return final_url
