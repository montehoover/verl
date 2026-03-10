import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_redirect_url(base_url_api, desired_endpoint, optional_params=None):
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.
    
    Args:
        base_url_api: string representing the main API base URL
        desired_endpoint: string representing the desired API endpoint
        optional_params: an optional dictionary for query string parameters
    
    Returns:
        Produces a complete API redirect URL as a string
    
    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint
    """
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url_api)
    domain = parsed_url.netloc
    
    # Validate the domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    # Validate the endpoint
    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not in the allowed endpoints list")
    
    # Construct the full URL
    full_url = base_url_api.rstrip('/') + desired_endpoint
    
    # Add query parameters if provided
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
