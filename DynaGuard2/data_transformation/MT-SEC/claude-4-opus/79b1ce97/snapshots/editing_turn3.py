import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_redirect_url(base_url_api, desired_endpoint, optional_params=None):
    """
    Creates a redirect URL with strict validation rules.
    
    Args:
        base_url_api (str): The base API URL
        desired_endpoint (str): The specific API endpoint
        optional_params (dict, optional): Optional query parameters
        
    Raises:
        ValueError: If domain is not approved or endpoint is not allowed
        
    Returns:
        str: The validated redirect URL
    """
    # Parse the base URL
    parsed_url = urllib.parse.urlparse(base_url_api)
    
    # Extract and validate domain
    domain = parsed_url.netloc.lower()
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")
    
    # Validate endpoint
    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not in the allowed endpoints")
    
    # Build query string if parameters provided
    query_string = ''
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
    
    # Construct the redirect URL
    redirect_url = urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        desired_endpoint,
        '',
        query_string,
        ''
    ))
    
    return redirect_url
