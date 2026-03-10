import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_domain(domain):
    """
    Validate if the domain is in the approved list.
    
    Args:
        domain: The domain to validate.
    
    Raises:
        ValueError: If the domain is not approved.
    """
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")

def validate_endpoint(api_path):
    """
    Validate if the API path is in the allowed endpoints.
    
    Args:
        api_path: The API endpoint path to validate.
    
    Raises:
        ValueError: If the endpoint is not allowed.
    """
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not in the list of allowed endpoints")

def construct_url(base_url, api_path, params=None):
    """
    Construct a complete URL from base URL, API path, and optional parameters.
    
    Args:
        base_url: The base URL.
        api_path: The API endpoint path.
        params: Optional dictionary of query parameters.
    
    Returns:
        The constructed URL as a string.
    """
    full_url = urllib.parse.urljoin(base_url, api_path)
    
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url

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
        ValueError: If the generated URL does not belong to an allowed domain or exceeds the list of allowed endpoints.
    """
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Validate domain and endpoint
    validate_domain(domain)
    validate_endpoint(api_path)
    
    # Construct and return the full URL
    return construct_url(base_url, api_path, params)
