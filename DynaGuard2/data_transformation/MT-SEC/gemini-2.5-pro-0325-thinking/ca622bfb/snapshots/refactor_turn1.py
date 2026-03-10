import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication.

    Args:
        root_url: The main URL of the application.
        path_for_callback: The path to execute OAuth callbacks.
        session_token: A string used to pass a state parameter for verifying the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or unauthorized domain.
    """
    parsed_root_url = urllib.parse.urlparse(root_url)
    if parsed_root_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {parsed_root_url.hostname} is not an allowed callback domain.")

    # Ensure path_for_callback starts with a slash if root_url doesn't end with one,
    # and path_for_callback is not empty.
    if not root_url.endswith('/') and path_for_callback and not path_for_callback.startswith('/'):
        full_path = root_url + '/' + path_for_callback
    else:
        # Handles cases where root_url ends with '/' or path_for_callback starts with '/' or is empty
        full_path = root_url.rstrip('/') + '/' + path_for_callback.lstrip('/')
    
    # Ensure full_path does not have double slashes if path_for_callback was empty or just "/"
    # This can happen if root_url ends with / and path_for_callback is / or empty.
    # A more robust way is to use urljoin.
    base_callback_url = urllib.parse.urljoin(root_url + ('/' if not root_url.endswith('/') else ''), path_for_callback.lstrip('/'))


    # Add the state parameter
    params = {'state': session_token}
    query_string = urllib.parse.urlencode(params)
    
    return f"{base_callback_url}?{query_string}"
