import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_link(base_url, path, params=None):
    # Parse the base URL to extract domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not approved")
    
    # Validate endpoint
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not allowed")
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(base_url, path)
    
    # Add query parameters if provided
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
