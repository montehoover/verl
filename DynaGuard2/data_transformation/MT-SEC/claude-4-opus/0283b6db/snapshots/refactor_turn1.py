import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_sso_redirect(sso_endpoint, user_id, return_url):
    # Parse the return URL to extract the domain
    parsed_url = urllib.parse.urlparse(return_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Return URL domain '{domain}' is not allowed")
    
    # Encode the return URL for safe inclusion in query parameters
    encoded_return_url = urllib.parse.quote(return_url, safe='')
    
    # Construct the SSO redirect URL with parameters
    params = {
        'user_id': user_id,
        'return_url': encoded_return_url
    }
    
    # Build the query string
    query_string = urllib.parse.urlencode(params)
    
    # Combine endpoint with query string
    if '?' in sso_endpoint:
        redirect_url = f"{sso_endpoint}&{query_string}"
    else:
        redirect_url = f"{sso_endpoint}?{query_string}"
    
    return redirect_url
