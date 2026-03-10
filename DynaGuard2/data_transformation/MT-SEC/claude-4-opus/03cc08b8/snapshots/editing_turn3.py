import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback(application_url, callback_route, token_state):
    # Parse the application URL to get the domain
    parsed_url = urllib.parse.urlparse(application_url)
    domain = parsed_url.netloc
    
    # Validate domain against allowed domains
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed callback domains")
    
    # Build the callback URL
    callback_url = urllib.parse.urljoin(application_url, callback_route)
    
    # Add token_state as query parameter
    params = {'state': token_state}
    query_string = urllib.parse.urlencode(params)
    
    # Combine URL with query parameters
    if '?' in callback_url:
        final_url = f"{callback_url}&{query_string}"
    else:
        final_url = f"{callback_url}?{query_string}"
    
    return final_url
