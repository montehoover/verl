import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_callback_domain(callback_domain, host_domain, allowed_domains):
    """Validate that the callback domain is either the same as host or in allowed list."""
    if callback_domain != host_domain and callback_domain not in allowed_domains:
        raise ValueError(f"Callback domain '{callback_domain}' is not allowed")
    return True

def construct_callback_url(host_url, path_callback):
    """Construct the full callback URL from host URL and callback path."""
    parsed_callback = urllib.parse.urlparse(path_callback)
    
    if parsed_callback.netloc:
        # Callback is already an absolute URL
        return path_callback
    else:
        # Callback is a relative path, combine with host URL
        return urllib.parse.urljoin(host_url, path_callback)

def add_state_to_url(url, state_value):
    """Add or update the state parameter in the URL query string."""
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    query_params['state'] = [state_value]
    
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))

def generate_oauth_url(host_url, path_callback, session_id):
    """Generate OAuth callback URL with state parameter."""
    # Extract domains for validation
    parsed_host = urllib.parse.urlparse(host_url)
    host_domain = parsed_host.netloc
    
    # Construct the callback URL
    callback_url = construct_callback_url(host_url, path_callback)
    
    # Validate the callback domain
    parsed_callback_url = urllib.parse.urlparse(callback_url)
    callback_domain = parsed_callback_url.netloc
    validate_callback_domain(callback_domain, host_domain, ALLOWED_CALLBACK_DOMAINS)
    
    # Add state parameter and return
    return add_state_to_url(callback_url, session_id)
