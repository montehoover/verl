import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def build_sso_url(sso_path, uid, redirect_url):
    # Parse the redirect URL to extract the domain
    parsed_url = urllib.parse.urlparse(redirect_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    # Build the SSO URL with query parameters
    params = {
        'uid': uid,
        'redirect': redirect_url
    }
    
    # Encode the parameters
    query_string = urllib.parse.urlencode(params)
    
    # Combine the SSO path with the query string
    if '?' in sso_path:
        # If sso_path already has query parameters, append with &
        sso_url = f"{sso_path}&{query_string}"
    else:
        # Otherwise, append with ?
        sso_url = f"{sso_path}?{query_string}"
    
    return sso_url
