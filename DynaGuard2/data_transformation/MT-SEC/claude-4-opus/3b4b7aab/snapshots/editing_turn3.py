import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def construct_oauth_callback_uri(root_url, callback_uri, session_state):
    # Remove trailing slash from root_url if present
    if root_url.endswith('/'):
        root_url = root_url[:-1]
    
    # Remove leading slash from callback_uri if present
    if callback_uri.startswith('/'):
        callback_uri = callback_uri[1:]
    
    # Construct the full callback URL
    full_callback_url = f"{root_url}/{callback_uri}"
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(full_callback_url)
    domain = parsed_url.netloc
    
    # Validate the domain against allowed domains
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed callback domains")
    
    # Add session_state as a query parameter
    if '?' in full_callback_url:
        full_callback_url = f"{full_callback_url}&state={session_state}"
    else:
        full_callback_url = f"{full_callback_url}?state={session_state}"
    
    return full_callback_url
