import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_redirect_url(main_url, endpoint_name, params_dict=None):
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.
    
    Args:
        main_url: string representing the main API base URL
        endpoint_name: string representing the desired API endpoint
        params_dict: an optional dictionary for query string parameters
    
    Returns:
        Produces a complete API redirect URL as a string
    
    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint
    """
    # Parse the main URL to extract the domain
    parsed_url = urllib.parse.urlparse(main_url)
    domain = parsed_url.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not in the allowed endpoints list")
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(main_url, endpoint_name)
    
    # Add query parameters if provided
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
