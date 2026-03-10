import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def extract_domain(parsed_url):
    """Extract domain from parsed URL, handling cases with port numbers."""
    return parsed_url.netloc.split(':')[0] if ':' in parsed_url.netloc else parsed_url.netloc

def validate_domain(domain):
    """Validate that the domain is in the approved list."""
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in approved domains")
    return domain

def validate_endpoint(endpoint):
    """Validate that the endpoint is in the allowed list."""
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in allowed endpoints")
    return endpoint

def build_base_url(parsed_url, endpoint):
    """Construct the base URL with the given endpoint."""
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        endpoint,
        '',
        '',
        ''
    ))

def append_query_params(base_url, params_dict):
    """Append query parameters to the base URL if provided."""
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{base_url}?{query_string}"
    return base_url

def construct_redirect_url(main_url, endpoint_name, params_dict=None):
    # Pipeline of operations
    parsed_url = urllib.parse.urlparse(main_url)
    
    # Extract and validate domain
    domain = extract_domain(parsed_url)
    validate_domain(domain)
    
    # Validate endpoint
    validate_endpoint(endpoint_name)
    
    # Build URL
    base_url = build_base_url(parsed_url, endpoint_name)
    
    # Add query parameters
    final_url = append_query_params(base_url, params_dict)
    
    return final_url
