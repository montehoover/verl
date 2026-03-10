import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url, redirect_path, nonce):
    """
    Builds an OAuth callback URL for third-party authentication.
    
    Args:
        main_url: a string indicating the main URL of the application.
        redirect_path: a string specifying the path to execute OAuth callbacks.
        nonce: a string used to pass a state parameter for verifying the OAuth transaction.
    
    Returns:
        Returns the full OAuth callback URL as a string.
    
    Raises:
        ValueError: raised if the callback URL points to an invalid or unauthorized domain.
    """
    # Parse the main URL to extract its components
    parsed_url = urllib.parse.urlparse(main_url)
    
    # Check if the domain is in the allowed list
    if parsed_url.netloc not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Invalid or unauthorized domain: {parsed_url.netloc}")
    
    # Ensure redirect_path starts with /
    if not redirect_path.startswith('/'):
        redirect_path = '/' + redirect_path
    
    # Build the base callback URL
    callback_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        redirect_path,
        '',
        '',
        ''
    ))
    
    # Add the state parameter
    params = {'state': nonce}
    final_url = callback_url + '?' + urllib.parse.urlencode(params)
    
    return final_url
