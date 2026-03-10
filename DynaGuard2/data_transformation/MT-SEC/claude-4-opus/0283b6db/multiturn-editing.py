import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_user_url(base_url, user_id):
    if '?' in base_url:
        return f"{base_url}&user_id={user_id}"
    else:
        return f"{base_url}?user_id={user_id}"

def generate_user_redirect_url(base_url, user_id, return_url):
    if '?' in base_url:
        return f"{base_url}&user_id={user_id}&return_url={return_url}"
    else:
        return f"{base_url}?user_id={user_id}&return_url={return_url}"

def generate_sso_redirect(sso_endpoint, user_id, return_url):
    # Parse the return URL to extract the domain
    parsed_url = urllib.parse.urlparse(return_url)
    domain = parsed_url.netloc
    
    # Validate the domain against allowed domains
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Return URL domain '{domain}' is not allowed")
    
    # URL encode the return_url to safely include it as a query parameter
    encoded_return_url = urllib.parse.quote(return_url, safe='')
    
    # Build the SSO redirect URL
    if '?' in sso_endpoint:
        return f"{sso_endpoint}&user_id={user_id}&return_url={encoded_return_url}"
    else:
        return f"{sso_endpoint}?user_id={user_id}&return_url={encoded_return_url}"
