import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_domain(domain):
    """
    Validates if the domain is in the approved API domains.
    
    Args:
        domain: The domain to validate.
    
    Raises:
        ValueError: If the domain is not approved.
    """
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")

def validate_endpoint(endpoint):
    """
    Validates if the endpoint is in the allowed endpoints.
    
    Args:
        endpoint: The endpoint to validate.
    
    Raises:
        ValueError: If the endpoint is not allowed.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the allowed endpoints")

def add_query_parameters(url, query_params):
    """
    Adds query parameters to a URL.
    
    Args:
        url: The base URL to add parameters to.
        query_params: Dictionary of query parameters to add.
    
    Returns:
        The URL with query parameters appended.
    """
    if not query_params:
        return url
    
    query_string = urllib.parse.urlencode(query_params)
    return f"{url}?{query_string}"

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
    
    # Validate domain and endpoint
    validate_domain(domain)
    validate_endpoint(endpoint)
    
    # Construct the full URL
    full_url = urllib.parse.urljoin(api_base_url, endpoint)
    
    # Add query parameters if provided
    return add_query_parameters(full_url, query_params)
