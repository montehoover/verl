import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def generate_api_redirect(base_url, api_path, params=None):
    """
    Generate a redirect URL for making API requests.
    
    Args:
        base_url: A string representing the base address of an API.
        api_path: A string specifying the endpoint of the API.
        params: An optional dictionary containing the query parameters.
    
    Returns:
        A fully constructed API redirect URL as a string.
    
    Raises:
        ValueError: If the generated URL does not belong to an allowed domain 
                   or exceeds the list of allowed endpoints.
    """
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")
    
    # Validate endpoint
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not in the list of allowed endpoints")
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(base_url, api_path)
    
    # Add query parameters if provided
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url
