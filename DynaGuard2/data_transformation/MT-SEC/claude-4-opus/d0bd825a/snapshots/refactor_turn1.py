import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url_api)
    domain = parsed_url.netloc
    
    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")
    
    # Check if the endpoint is in the allowed list
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not in the list of allowed endpoints")
    
    # Combine base URL and endpoint
    # Remove trailing slash from base_url if present and ensure endpoint starts with /
    base_url_clean = base_url_api.rstrip('/')
    endpoint_clean = api_endpoint if api_endpoint.startswith('/') else '/' + api_endpoint
    
    full_url = base_url_clean + endpoint_clean
    
    # Add query parameters if provided
    if opts:
        query_string = urllib.parse.urlencode(opts)
        full_url = full_url + '?' + query_string
    
    return full_url
