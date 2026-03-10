import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def extract_domain(base_url_api):
    """Extract domain from a URL."""
    parsed_url = urllib.parse.urlparse(base_url_api)
    return parsed_url.netloc

def validate_domain(domain):
    """Validate if domain is in approved list."""
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")

def validate_endpoint(endpoint):
    """Validate if endpoint is in allowed list."""
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the allowed endpoints list")

def normalize_url_parts(base_url_api, desired_endpoint):
    """Normalize URL parts by handling trailing/leading slashes."""
    base_url_normalized = base_url_api.rstrip('/')
    endpoint_normalized = desired_endpoint if desired_endpoint.startswith('/') else '/' + desired_endpoint
    return base_url_normalized, endpoint_normalized

def construct_url(base_url_api, desired_endpoint, optional_params=None):
    """Construct the full URL with optional query parameters."""
    base_url_normalized, endpoint_normalized = normalize_url_parts(base_url_api, desired_endpoint)
    full_url = base_url_normalized + endpoint_normalized
    
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        full_url = full_url + '?' + query_string
    
    return full_url

def create_redirect_url(base_url_api, desired_endpoint, optional_params=None):
    """
    Creates a full redirect URL for API requests by combining the base API URL, 
    a specific endpoint, and optional query parameters.
    
    Args:
        base_url_api: string representing the main API base URL.
        desired_endpoint: string representing the desired API endpoint.
        optional_params: an optional dictionary for query string parameters.
    
    Returns:
        Produces a complete API redirect URL as a string.
    
    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint.
    """
    # Extract and validate domain
    domain = extract_domain(base_url_api)
    validate_domain(domain)
    
    # Validate endpoint
    validate_endpoint(desired_endpoint)
    
    # Construct and return the full URL
    return construct_url(base_url_api, desired_endpoint, optional_params)
