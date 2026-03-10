import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_redirect_url(main_url, endpoint_name, params_dict=None):
    # Parse the main URL
    parsed_url = urllib.parse.urlparse(main_url)
    
    # Extract domain from netloc (handles cases with port numbers)
    domain = parsed_url.netloc.split(':')[0] if ':' in parsed_url.netloc else parsed_url.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in approved domains")
    
    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not in allowed endpoints")
    
    # Construct the base URL with endpoint
    base_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        endpoint_name,
        '',
        '',
        ''
    ))
    
    # Add query parameters if provided
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{base_url}?{query_string}"
    
    return base_url
