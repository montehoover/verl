import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_callback_domain(callback_domain, app_domain, allowed_domains):
    """Validate that the callback domain is either the same as app domain or in allowed list."""
    if callback_domain != app_domain and callback_domain not in allowed_domains:
        raise ValueError(f"Callback domain '{callback_domain}' is not allowed")
    return True

def construct_callback_url(application_url, callback_route):
    """Construct the full callback URL from application URL and callback route."""
    parsed_callback = urllib.parse.urlparse(callback_route)
    
    if parsed_callback.netloc:
        # callback_route contains a domain, validate it
        parsed_app = urllib.parse.urlparse(application_url)
        validate_callback_domain(parsed_callback.netloc, parsed_app.netloc, ALLOWED_CALLBACK_DOMAINS)
        return callback_route
    else:
        # callback_route is just a path, combine with application URL
        return urllib.parse.urljoin(application_url, callback_route)

def add_state_parameter(url, state_value):
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

def assemble_oauth_callback(application_url, callback_route, token_state):
    """Assemble a complete OAuth callback URL with state parameter."""
    # Pipeline: construct URL -> add state parameter
    callback_url = construct_callback_url(application_url, callback_route)
    final_url = add_state_parameter(callback_url, token_state)
    return final_url
