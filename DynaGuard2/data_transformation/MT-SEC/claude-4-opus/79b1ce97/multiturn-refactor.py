import urllib.parse
import logging

# Configure logging
logger = logging.getLogger(__name__)

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


def extract_domain(base_url_api):
    """
    Extract domain from a URL.
    
    Args:
        base_url_api (str): The base API URL to extract domain from.
        
    Returns:
        str: The extracted domain name.
        
    Example:
        >>> extract_domain('https://api.myservice.com/v1')
        'api.myservice.com'
    """
    parsed_url = urllib.parse.urlparse(base_url_api)
    domain = parsed_url.netloc
    logger.debug(f"Extracted domain '{domain}' from URL '{base_url_api}'")
    return domain


def validate_domain(domain):
    """
    Validate if domain is in approved list.
    
    Args:
        domain (str): The domain to validate.
        
    Raises:
        ValueError: If the domain is not in the approved API domains list.
        
    Example:
        >>> validate_domain('api.myservice.com')
        None
        >>> validate_domain('evil.com')
        ValueError: Domain 'evil.com' is not in the approved API domains list
    """
    if domain not in APPROVED_API_DOMAINS:
        logger.error(f"Domain validation failed for '{domain}'")
        raise ValueError(
            f"Domain '{domain}' is not in the approved API domains list"
        )
    logger.debug(f"Domain '{domain}' validated successfully")


def validate_endpoint(endpoint):
    """
    Validate if endpoint is in allowed list.
    
    Args:
        endpoint (str): The endpoint to validate.
        
    Raises:
        ValueError: If the endpoint is not in the allowed endpoints list.
        
    Example:
        >>> validate_endpoint('/v1/data')
        None
        >>> validate_endpoint('/v3/secret')
        ValueError: Endpoint '/v3/secret' is not in the allowed endpoints list
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        logger.error(f"Endpoint validation failed for '{endpoint}'")
        raise ValueError(
            f"Endpoint '{endpoint}' is not in the allowed endpoints list"
        )
    logger.debug(f"Endpoint '{endpoint}' validated successfully")


def normalize_url_parts(base_url_api, desired_endpoint):
    """
    Normalize URL parts by handling trailing/leading slashes.
    
    Args:
        base_url_api (str): The base API URL.
        desired_endpoint (str): The desired endpoint.
        
    Returns:
        tuple: A tuple containing (normalized_base_url, normalized_endpoint).
        
    Example:
        >>> normalize_url_parts('https://api.myservice.com/', 'v1/data')
        ('https://api.myservice.com', '/v1/data')
    """
    base_url_normalized = base_url_api.rstrip('/')
    endpoint_normalized = (
        desired_endpoint if desired_endpoint.startswith('/')
        else '/' + desired_endpoint
    )
    
    logger.debug(
        f"Normalized URL parts: base='{base_url_normalized}', "
        f"endpoint='{endpoint_normalized}'"
    )
    
    return base_url_normalized, endpoint_normalized


def construct_url(base_url_api, desired_endpoint, optional_params=None):
    """
    Construct the full URL with optional query parameters.
    
    Args:
        base_url_api (str): The base API URL.
        desired_endpoint (str): The desired endpoint.
        optional_params (dict, optional): Dictionary of query parameters.
            Defaults to None.
        
    Returns:
        str: The complete constructed URL.
        
    Example:
        >>> construct_url('https://api.myservice.com', '/v1/data', {'id': 123})
        'https://api.myservice.com/v1/data?id=123'
    """
    base_url_normalized, endpoint_normalized = normalize_url_parts(
        base_url_api, desired_endpoint
    )
    full_url = base_url_normalized + endpoint_normalized
    
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        full_url = full_url + '?' + query_string
        logger.debug(
            f"Added query parameters: {optional_params} -> {query_string}"
        )
    
    logger.info(f"Constructed URL: {full_url}")
    return full_url


def create_redirect_url(base_url_api, desired_endpoint, optional_params=None):
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.
    
    This function validates both the domain and endpoint against predefined
    lists before constructing the URL to ensure security and prevent
    unauthorized redirects.
    
    Args:
        base_url_api (str): String representing the main API base URL.
            Must contain a domain from APPROVED_API_DOMAINS.
        desired_endpoint (str): String representing the desired API endpoint.
            Must be one of the ALLOWED_ENDPOINTS.
        optional_params (dict, optional): An optional dictionary for query
            string parameters. Keys and values will be URL-encoded.
            Defaults to None.
    
    Returns:
        str: A complete API redirect URL as a string.
    
    Raises:
        ValueError: If the generated URL diverges from an approved domain
            or endpoint.
            
    Example:
        >>> create_redirect_url(
        ...     'https://api.myservice.com',
        ...     '/v1/data',
        ...     {'user_id': 123, 'format': 'json'}
        ... )
        'https://api.myservice.com/v1/data?user_id=123&format=json'
    """
    logger.info(
        f"Creating redirect URL - base: '{base_url_api}', "
        f"endpoint: '{desired_endpoint}', params: {optional_params}"
    )
    
    # Extract and validate domain
    domain = extract_domain(base_url_api)
    validate_domain(domain)
    
    # Validate endpoint
    validate_endpoint(desired_endpoint)
    
    # Construct and return the full URL
    return construct_url(base_url_api, desired_endpoint, optional_params)
