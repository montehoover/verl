import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def build_api_redirect_url(api_base_url, endpoint, query_params=None):
    """
    Constructs a redirect URL for API responses.
    
    Args:
        api_base_url: A string representing the base URL of the API.
        endpoint: A string representing the specific API endpoint.
        query_params: An optional dictionary of query parameters.
    
    Returns:
        A string that is the complete API redirect URL.
    
    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint.
    """
    # Parse the base URL to extract the domain
    parsed_base = urllib.parse.urlparse(api_base_url)
    domain = parsed_base.netloc
    
    # Check if the domain is approved
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")
    
    # Check if the endpoint is allowed
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the allowed endpoints")
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(api_base_url, endpoint)
    
    # Add query parameters if provided
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
