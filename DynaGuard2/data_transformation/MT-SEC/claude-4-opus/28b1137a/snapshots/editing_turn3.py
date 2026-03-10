import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def construct_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path

def build_user_url(base_url, path, user_id):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    return f"{base_url}{path}?user_id={user_id}"

def build_sso_url(sso_path, uid, redirect_url):
    # Parse the redirect URL to extract the domain
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    redirect_domain = parsed_redirect.netloc
    
    # Validate the redirect domain
    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain '{redirect_domain}' is not in the allowed list of return domains")
    
    # Build the SSO URL with query parameters
    params = {
        'uid': uid,
        'redirect_url': redirect_url
    }
    
    # Parse the SSO path and add query parameters
    parsed_sso = urllib.parse.urlparse(sso_path)
    query_params = urllib.parse.urlencode(params)
    
    # Construct the final URL
    sso_url = urllib.parse.urlunparse((
        parsed_sso.scheme,
        parsed_sso.netloc,
        parsed_sso.path,
        parsed_sso.params,
        query_params,
        parsed_sso.fragment
    ))
    
    return sso_url
