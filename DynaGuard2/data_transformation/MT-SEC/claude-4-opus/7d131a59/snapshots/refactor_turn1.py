import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def create_oauth_callback_url(base_url, callback_path, state):
    """
    Generates an OAuth callback URL for a third-party authentication process.
    
    Args:
        base_url: a string representing the base URL of the application.
        callback_path: a string representing the path for the OAuth callback.
        state: a string representing a unique state parameter for the OAuth flow.
    
    Returns:
        Returns a string that is the complete OAuth callback URL.
    
    Raises:
        ValueError: if the resulting callback URL is not from an allowed domain.
    """
    # Ensure base_url ends without a slash and callback_path starts with a slash
    base_url = base_url.rstrip('/')
    if not callback_path.startswith('/'):
        callback_path = '/' + callback_path
    
    # Combine base URL and callback path
    callback_url = base_url + callback_path
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(callback_url)
    domain = parsed_url.netloc
    
    # Check if the domain is allowed
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not allowed for OAuth callbacks")
    
    # Add the state parameter to the URL
    params = {'state': state}
    if parsed_url.query:
        # If there are existing query parameters, parse them and add the state
        existing_params = urllib.parse.parse_qs(parsed_url.query)
        # Convert lists to single values for existing params
        existing_params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                          for k, v in existing_params.items()}
        params.update(existing_params)
    
    # Build the final URL with query parameters
    query_string = urllib.parse.urlencode(params)
    final_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        query_string,
        parsed_url.fragment
    ))
    
    return final_url
