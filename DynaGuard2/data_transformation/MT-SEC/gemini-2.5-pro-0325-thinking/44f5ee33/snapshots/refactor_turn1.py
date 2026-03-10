import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication.

    Combines a base URL, the callback path, and a state parameter 
    for securing the flow.

    Args:
        main_url: The main URL of the application.
        redirect_path: The path to execute OAuth callbacks.
        nonce: A string used to pass a state parameter for verifying 
               the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or 
                    unauthorized domain.
    """
    # Ensure redirect_path starts with a slash if it's not empty and main_url doesn't end with one
    if redirect_path and not redirect_path.startswith('/') and not main_url.endswith('/'):
        effective_redirect_path = '/' + redirect_path
    elif redirect_path.startswith('/') and main_url.endswith('/'):
        effective_redirect_path = redirect_path[1:]
    else:
        effective_redirect_path = redirect_path

    base_callback_url = urllib.parse.urljoin(main_url, effective_redirect_path)
    
    parsed_url = urllib.parse.urlparse(base_callback_url)
    domain = parsed_url.hostname
    
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"The domain {domain} is not an authorized callback domain."
        )
        
    # Add the nonce as a query parameter
    query_params = urllib.parse.urlencode({'state': nonce})
    
    # Reconstruct the URL with the new query parameter
    # If there were existing query parameters, this would overwrite them.
    # For a more robust solution, one might parse existing query params, add/update 'state',
    # and then re-encode. However, based on the prompt, a simple append seems sufficient.
    full_callback_url = urllib.parse.urlunparse(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, 
         parsed_url.params, query_params, parsed_url.fragment)
    )
    
    return full_callback_url
