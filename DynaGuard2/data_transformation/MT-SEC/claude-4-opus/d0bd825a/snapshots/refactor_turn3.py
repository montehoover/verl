import urllib.parse

# Configuration constants for API security
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def extract_domain(base_url):
    """
    Extract the domain from a URL.
    
    Args:
        base_url (str): The base URL to extract domain from.
        
    Returns:
        str: The extracted domain (netloc) from the URL.
    """
    parsed_url = urllib.parse.urlparse(base_url)
    return parsed_url.netloc


def validate_domain(domain):
    """
    Validate if the domain is in the approved list.
    
    Args:
        domain (str): The domain to validate.
        
    Returns:
        str: The domain if valid.
        
    Raises:
        ValueError: If the domain is not in the approved list.
    """
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved API domains")
    return domain


def validate_endpoint(endpoint):
    """
    Validate if the endpoint is in the allowed list.
    
    Args:
        endpoint (str): The API endpoint to validate.
        
    Returns:
        str: The endpoint if valid.
        
    Raises:
        ValueError: If the endpoint is not in the allowed list.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the list of allowed endpoints")
    return endpoint


def clean_base_url(base_url):
    """
    Remove trailing slash from base URL to prevent double slashes.
    
    Args:
        base_url (str): The base URL to clean.
        
    Returns:
        str: The base URL without trailing slash.
    """
    return base_url.rstrip('/')


def normalize_endpoint(endpoint):
    """
    Ensure endpoint starts with a forward slash for consistent URL construction.
    
    Args:
        endpoint (str): The endpoint to normalize.
        
    Returns:
        str: The endpoint with a leading forward slash.
    """
    return endpoint if endpoint.startswith('/') else '/' + endpoint


def construct_url(base_url, endpoint):
    """
    Combine base URL and endpoint into a complete URL.
    
    Args:
        base_url (str): The base URL of the API.
        endpoint (str): The API endpoint.
        
    Returns:
        str: The complete URL combining base and endpoint.
    """
    clean_url = clean_base_url(base_url)
    normalized_endpoint = normalize_endpoint(endpoint)
    return clean_url + normalized_endpoint


def add_query_params(url, params):
    """
    Add query parameters to a URL if provided.
    
    Args:
        url (str): The URL to add parameters to.
        params (dict or None): Optional dictionary of query parameters.
        
    Returns:
        str: The URL with query parameters appended, or original URL if no params.
    """
    if params:
        query_string = urllib.parse.urlencode(params)
        return url + '?' + query_string
    return url


def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    """
    Generate a secure redirect URL for API requests with validation.
    
    This function creates a complete API URL by combining a base URL and endpoint,
    with optional query parameters. It ensures security by validating both the
    domain and endpoint against pre-approved lists.
    
    Args:
        base_url_api (str): The base address of the API (e.g., 'https://api.myservice.com').
        api_endpoint (str): The specific API endpoint (e.g., '/v1/data').
        opts (dict, optional): Dictionary containing query parameters. Defaults to None.
        
    Returns:
        str: A fully constructed and validated API redirect URL.
        
    Raises:
        ValueError: If the domain is not in APPROVED_API_DOMAINS or
                   if the endpoint is not in ALLOWED_ENDPOINTS.
                   
    Example:
        >>> create_api_redirect_url('https://api.myservice.com', '/v1/data', {'limit': 10})
        'https://api.myservice.com/v1/data?limit=10'
    """
    # Step 1: Extract and validate the domain for security
    domain = extract_domain(base_url_api)
    validate_domain(domain)
    
    # Step 2: Validate the endpoint against allowed list
    validate_endpoint(api_endpoint)
    
    # Step 3: Construct the base URL with endpoint
    url = construct_url(base_url_api, api_endpoint)
    
    # Step 4: Add query parameters if provided
    final_url = add_query_params(url, opts)
    
    return final_url
