import urllib.parse
from urllib.parse import urlparse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_redirect_url(main_url, endpoint_name, params_dict):
    # Parse the main URL
    parsed = urlparse(main_url)
    
    # Extract and validate domain
    domain = parsed.netloc.lower()
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved list")
    
    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not in the allowed list")
    
    # Construct the base URL with the endpoint
    base_url = f"{parsed.scheme}://{parsed.netloc}{endpoint_name}"
    
    # Add query parameters if provided
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{base_url}?{query_string}"
    
    return base_url
