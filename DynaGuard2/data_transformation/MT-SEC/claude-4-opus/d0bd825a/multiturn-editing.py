from urllib.parse import urlparse, urlencode, urlunparse
import urllib.parse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_API_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_url_domain(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        return False

def validate_url_with_endpoint(url, endpoint):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is approved
        if domain not in APPROVED_DOMAINS:
            return False
        
        # Check if endpoint is allowed
        if endpoint not in ALLOWED_ENDPOINTS:
            return False
        
        return True
    except Exception:
        return False

def create_api_redirect_url(base_url_api, api_endpoint, opts):
    # Parse the base URL
    parsed_url = urlparse(base_url_api)
    domain = parsed_url.netloc.lower()
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")
    
    # Validate endpoint
    if api_endpoint not in ALLOWED_API_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not in the allowed API endpoints")
    
    # Build the query string from opts
    query_string = urlencode(opts) if opts else ''
    
    # Construct the full URL
    full_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        api_endpoint,
        '',
        query_string,
        ''
    ))
    
    return full_url
