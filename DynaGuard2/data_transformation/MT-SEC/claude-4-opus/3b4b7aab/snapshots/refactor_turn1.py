import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def construct_oauth_callback_uri(root_url, callback_uri, session_state):
    """
    Constructs an OAuth callback URL used during third-party authentication.
    
    Args:
        root_url: A string defining the application's main base URL.
        callback_uri: A string that represents the specific path for the OAuth callback endpoint.
        session_state: A string to include a unique value for ensuring the integrity of the OAuth request flow.
    
    Returns:
        A string representing the finalized OAuth callback URL.
    
    Raises:
        ValueError: Triggered if the computed callback URL's domain is not within the list of allowed domains.
    """
    # Combine base URL with callback path
    if root_url.endswith('/') and callback_uri.startswith('/'):
        full_url = root_url[:-1] + callback_uri
    elif not root_url.endswith('/') and not callback_uri.startswith('/'):
        full_url = root_url + '/' + callback_uri
    else:
        full_url = root_url + callback_uri
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(full_url)
    domain = parsed_url.netloc
    
    # Check if domain is allowed
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of allowed callback domains")
    
    # Append state parameter
    if '?' in full_url:
        final_url = f"{full_url}&state={urllib.parse.quote(session_state)}"
    else:
        final_url = f"{full_url}?state={urllib.parse.quote(session_state)}"
    
    return final_url
