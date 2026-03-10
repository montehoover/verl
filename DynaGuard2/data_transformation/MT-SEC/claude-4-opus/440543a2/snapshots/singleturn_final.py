import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_link(base_url, path, params=None):
    """
    Create an API redirect URL by merging a base API URL with a defined endpoint and optional query parameters.
    
    Args:
        base_url: A string designating the API's base URL.
        path: A string identifying the target API endpoint.
        params: An optional dictionary containing key-value pairs used as query parameters.
    
    Returns:
        A string representing the final, assembled API URL.
    
    Raises:
        ValueError: Triggered if the constructed URL points to an unapproved domain or endpoint.
    """
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Check if the domain is approved
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Check if the endpoint path is allowed
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not in the allowed endpoints list")
    
    # Combine base URL and path
    full_url = urllib.parse.urljoin(base_url, path)
    
    # Add query parameters if provided
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
