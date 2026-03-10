import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def combine_url_parts(root_url, callback_uri):
    """
    Combines root URL and callback URI into a single URL.
    
    Args:
        root_url: The base URL
        callback_uri: The callback path
        
    Returns:
        Combined URL string
    """
    if root_url.endswith('/') and callback_uri.startswith('/'):
        return root_url[:-1] + callback_uri
    elif not root_url.endswith('/') and not callback_uri.startswith('/'):
        return root_url + '/' + callback_uri
    else:
        return root_url + callback_uri

def validate_domain(url, allowed_domains):
    """
    Validates that the URL's domain is in the allowed list.
    
    Args:
        url: The URL to validate
        allowed_domains: Set of allowed domains
        
    Returns:
        The domain if valid
        
    Raises:
        ValueError: If domain is not allowed
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in allowed_domains:
        raise ValueError(f"Domain '{domain}' is not in the list of allowed callback domains")
    
    return domain

def append_state_parameter(url, state):
    """
    Appends state parameter to URL.
    
    Args:
        url: The base URL
        state: The state value to append
        
    Returns:
        URL with state parameter appended
    """
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}state={urllib.parse.quote(state)}"

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
    # Pipeline: combine -> validate -> append state
    full_url = combine_url_parts(root_url, callback_uri)
    validate_domain(full_url, ALLOWED_CALLBACK_DOMAINS)
    final_url = append_state_parameter(full_url, session_state)
    
    return final_url
