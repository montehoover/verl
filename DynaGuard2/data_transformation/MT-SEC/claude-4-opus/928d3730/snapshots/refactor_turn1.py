import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def generate_oauth_url(host_url, path_callback, session_id):
    # Parse the host URL to extract domain
    parsed_host = urllib.parse.urlparse(host_url)
    host_domain = parsed_host.netloc
    
    # Parse the callback path to check if it's an absolute URL
    parsed_callback = urllib.parse.urlparse(path_callback)
    
    if parsed_callback.netloc:
        # Callback is an absolute URL, check if domain is allowed
        callback_domain = parsed_callback.netloc
        if callback_domain != host_domain and callback_domain not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError(f"Callback domain '{callback_domain}' is not allowed")
        callback_url = path_callback
    else:
        # Callback is a relative path, combine with host URL
        callback_url = urllib.parse.urljoin(host_url, path_callback)
    
    # Add state parameter
    parsed_url = urllib.parse.urlparse(callback_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    query_params['state'] = [session_id]
    
    # Reconstruct URL with updated query parameters
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    oauth_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))
    
    return oauth_url
