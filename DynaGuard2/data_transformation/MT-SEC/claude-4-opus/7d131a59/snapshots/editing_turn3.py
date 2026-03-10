import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def create_oauth_callback_url(base_url, callback_path, state):
    if not base_url.startswith("https://"):
        raise ValueError("Base URL must use HTTPS protocol")
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {domain} is not in the allowed callback domains")
    
    # Combine base_url and callback_path
    if base_url.endswith("/") and callback_path.startswith("/"):
        url = base_url + callback_path[1:]
    elif not base_url.endswith("/") and not callback_path.startswith("/"):
        url = base_url + "/" + callback_path
    else:
        url = base_url + callback_path
    
    # Add state as query parameter
    if "?" in url:
        url += f"&state={state}"
    else:
        url += f"?state={state}"
    
    return url
