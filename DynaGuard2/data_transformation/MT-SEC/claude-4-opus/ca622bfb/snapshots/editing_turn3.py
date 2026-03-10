import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def assemble_oauth_callback_url(root_url, path_for_callback, session_token):
    # Parse the root_url to extract domain
    parsed_url = urllib.parse.urlparse(root_url)
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {domain} is not in the allowed callback domains")
    
    # Remove trailing slash from root_url if present
    if root_url.endswith('/'):
        root_url = root_url[:-1]
    
    # Ensure path_for_callback starts with /
    if not path_for_callback.startswith('/'):
        path_for_callback = '/' + path_for_callback
    
    # Combine URL and add state parameter
    return f"{root_url}{path_for_callback}?state={session_token}"
