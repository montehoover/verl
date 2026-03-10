import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback(application_url, callback_route, token_state):
    # Parse the application URL to extract its domain
    parsed_url = urllib.parse.urlparse(application_url)
    app_domain = parsed_url.netloc
    
    # Parse the callback route to check if it contains a domain
    parsed_callback = urllib.parse.urlparse(callback_route)
    
    # If callback_route contains a domain, validate it
    if parsed_callback.netloc:
        callback_domain = parsed_callback.netloc
        # Check if the callback domain matches the application domain or is in allowed domains
        if callback_domain != app_domain and callback_domain not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError(f"Callback domain '{callback_domain}' is not allowed")
        # Use the full callback URL as provided
        callback_url = callback_route
    else:
        # If callback_route is just a path, combine it with the application URL
        callback_url = urllib.parse.urljoin(application_url, callback_route)
    
    # Parse the callback URL and add the state parameter
    parsed_callback_url = urllib.parse.urlparse(callback_url)
    query_params = urllib.parse.parse_qs(parsed_callback_url.query)
    query_params['state'] = [token_state]
    
    # Reconstruct the URL with the state parameter
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    final_url = urllib.parse.urlunparse((
        parsed_callback_url.scheme,
        parsed_callback_url.netloc,
        parsed_callback_url.path,
        parsed_callback_url.params,
        new_query,
        parsed_callback_url.fragment
    ))
    
    return final_url
