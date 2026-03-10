import urllib.parse


# Constants for API security and validation
APPROVED_API_DOMAINS = {
    'api.myservice.com', 
    'api-test.myservice.com', 
    'api-staging.myservice.com'
}

ALLOWED_ENDPOINTS = [
    '/v1/data', 
    '/v1/user', 
    '/v2/analytics', 
    '/health'
]


def validate_domain(domain):
    """
    Validate if the domain is in the approved list.
    
    This function ensures that only pre-approved domains can be used
    for API requests, preventing potential security risks from
    arbitrary external domains.
    
    Args:
        domain (str): The domain to validate (e.g., 'api.myservice.com').
    
    Raises:
        ValueError: If the domain is not in the APPROVED_API_DOMAINS set.
    
    Example:
        >>> validate_domain('api.myservice.com')  # No exception
        >>> validate_domain('malicious.com')  # Raises ValueError
    """
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not in the list of approved API domains"
        )


def validate_endpoint(api_path):
    """
    Validate if the API path is in the allowed endpoints.
    
    This function ensures that only specific, pre-defined endpoints
    can be accessed, providing an additional layer of security by
    restricting the available API surface.
    
    Args:
        api_path (str): The API endpoint path to validate (e.g., '/v1/data').
    
    Raises:
        ValueError: If the endpoint is not in the ALLOWED_ENDPOINTS list.
    
    Example:
        >>> validate_endpoint('/v1/data')  # No exception
        >>> validate_endpoint('/v3/secret')  # Raises ValueError
    """
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(
            f"Endpoint '{api_path}' is not in the list of allowed endpoints"
        )


def construct_url(base_url, api_path, params=None):
    """
    Construct a complete URL from base URL, API path, and optional parameters.
    
    This pure function handles the URL construction logic, combining the
    base URL with the API path and encoding any query parameters in a
    URL-safe format.
    
    Args:
        base_url (str): The base URL (e.g., 'https://api.myservice.com').
        api_path (str): The API endpoint path (e.g., '/v1/data').
        params (dict, optional): Dictionary of query parameters to append
            to the URL. Defaults to None.
    
    Returns:
        str: The constructed URL as a string, with properly encoded
            query parameters if provided.
    
    Example:
        >>> construct_url('https://api.myservice.com', '/v1/data')
        'https://api.myservice.com/v1/data'
        
        >>> construct_url('https://api.myservice.com', '/v1/data', 
        ...               {'key': 'value', 'limit': 10})
        'https://api.myservice.com/v1/data?key=value&limit=10'
    """
    # Join base URL and API path, handling trailing/leading slashes
    full_url = urllib.parse.urljoin(base_url, api_path)
    
    # Append query parameters if provided
    if params:
        # Encode parameters to handle special characters properly
        query_string = urllib.parse.urlencode(params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url


def generate_api_redirect(base_url, api_path, params=None):
    """
    Generate a redirect URL for making API requests.
    
    This is the main function that orchestrates the URL generation process.
    It validates both the domain and endpoint against pre-approved lists
    before constructing the final redirect URL. This ensures that only
    safe, authorized API calls can be made.
    
    Args:
        base_url (str): A string representing the base address of an API
            (e.g., 'https://api.myservice.com').
        api_path (str): A string specifying the endpoint of the API
            (e.g., '/v1/data').
        params (dict, optional): An optional dictionary containing the query
            parameters to be appended to the URL. Defaults to None.
    
    Returns:
        str: A fully constructed API redirect URL as a string.
    
    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
            or exceeds the list of allowed endpoints.
    
    Example:
        >>> generate_api_redirect('https://api.myservice.com', '/v1/data')
        'https://api.myservice.com/v1/data'
        
        >>> generate_api_redirect('https://api.myservice.com', '/v1/user',
        ...                       {'id': '123', 'format': 'json'})
        'https://api.myservice.com/v1/user?id=123&format=json'
        
        >>> generate_api_redirect('https://evil.com', '/v1/data')
        ValueError: Domain 'evil.com' is not in the list of approved API domains
    """
    # Parse the base URL to extract the domain for validation
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    
    # Perform security validations
    validate_domain(domain)
    validate_endpoint(api_path)
    
    # Construct and return the validated URL
    return construct_url(base_url, api_path, params)
