import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback_url(root_url, path_for_callback, session_token):
    """
    Builds an OAuth callback URL for third-party authentication.
    
    Args:
        root_url: a string indicating the main URL of the application.
        path_for_callback: a string specifying the path to execute OAuth callbacks.
        session_token: a string used to pass a state parameter for verifying the OAuth transaction.
    
    Returns:
        Returns the full OAuth callback URL as a string.
    
    Raises:
        ValueError: raised if the callback URL points to an invalid or unauthorized domain.
    """
    # Parse the root URL to extract the domain
    parsed_url = urllib.parse.urlparse(root_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not authorized for OAuth callbacks")
    
    # Ensure the path starts with a slash
    if not path_for_callback.startswith('/'):
        path_for_callback = '/' + path_for_callback
    
    # Build the full callback URL
    callback_url = urllib.parse.urljoin(root_url, path_for_callback)
    
    # Add the state parameter
    parsed_callback = urllib.parse.urlparse(callback_url)
    query_params = urllib.parse.parse_qs(parsed_callback.query)
    query_params['state'] = [session_token]
    
    # Reconstruct the URL with the state parameter
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    final_url = urllib.parse.urlunparse((
        parsed_callback.scheme,
        parsed_callback.netloc,
        parsed_callback.path,
        parsed_callback.params,
        new_query,
        parsed_callback.fragment
    ))
    
    return final_url
