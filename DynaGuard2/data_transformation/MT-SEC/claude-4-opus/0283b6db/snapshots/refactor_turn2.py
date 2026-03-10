import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def extract_domain(url):
    """Extract domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc

def validate_domain(domain, allowed_domains):
    """Validate if domain is in allowed list."""
    if domain not in allowed_domains:
        raise ValueError(f"Return URL domain '{domain}' is not allowed")
    return domain

def build_sso_params(user_id, return_url):
    """Build SSO query parameters."""
    return {
        'user_id': user_id,
        'return_url': urllib.parse.quote(return_url, safe='')
    }

def construct_redirect_url(endpoint, params):
    """Construct the final redirect URL with parameters."""
    query_string = urllib.parse.urlencode(params)
    separator = '&' if '?' in endpoint else '?'
    return f"{endpoint}{separator}{query_string}"

def generate_sso_redirect(sso_endpoint, user_id, return_url):
    # Pipeline: extract domain -> validate -> build params -> construct URL
    domain = extract_domain(return_url)
    validate_domain(domain, ALLOWED_RETURN_DOMAINS)
    params = build_sso_params(user_id, return_url)
    redirect_url = construct_redirect_url(sso_endpoint, params)
    
    return redirect_url
