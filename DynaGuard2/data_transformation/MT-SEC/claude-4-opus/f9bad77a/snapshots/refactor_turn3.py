import urllib.parse

# Configuration constants
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def extract_domain(parsed_url):
    """Extract domain from parsed URL, handling cases with port numbers.
    
    Args:
        parsed_url: A ParseResult object from urllib.parse.urlparse()
        
    Returns:
        str: The domain name without port number
    """
    return parsed_url.netloc.split(':')[0] if ':' in parsed_url.netloc else parsed_url.netloc


def validate_domain(domain):
    """Validate that the domain is in the approved list.
    
    Args:
        domain (str): The domain name to validate
        
    Returns:
        str: The validated domain name
        
    Raises:
        ValueError: If the domain is not in APPROVED_API_DOMAINS
    """
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in approved domains")
    return domain


def validate_endpoint(endpoint):
    """Validate that the endpoint is in the allowed list.
    
    Args:
        endpoint (str): The API endpoint path to validate
        
    Returns:
        str: The validated endpoint path
        
    Raises:
        ValueError: If the endpoint is not in ALLOWED_ENDPOINTS
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in allowed endpoints")
    return endpoint


def build_base_url(parsed_url, endpoint):
    """Construct the base URL with the given endpoint.
    
    Args:
        parsed_url: A ParseResult object from urllib.parse.urlparse()
        endpoint (str): The API endpoint path
        
    Returns:
        str: The complete base URL with scheme, domain, and endpoint
    """
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        endpoint,
        '',
        '',
        ''
    ))


def append_query_params(base_url, params_dict):
    """Append query parameters to the base URL if provided.
    
    Args:
        base_url (str): The base URL to append parameters to
        params_dict (dict, optional): Dictionary of query parameters
        
    Returns:
        str: The URL with query parameters appended, or original URL if no parameters
    """
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{base_url}?{query_string}"
    return base_url


def construct_redirect_url(main_url, endpoint_name, params_dict=None):
    """Create a full redirect URL for API requests.
    
    Combines the base API URL, a specific endpoint, and optional query parameters
    into a complete, validated API redirect URL.
    
    Args:
        main_url (str): The main API base URL
        endpoint_name (str): The desired API endpoint path
        params_dict (dict, optional): Dictionary for query string parameters
        
    Returns:
        str: A complete API redirect URL as a string
        
    Raises:
        ValueError: If the generated URL diverges from an approved domain or endpoint
    """
    # Parse the input URL
    parsed_url = urllib.parse.urlparse(main_url)
    
    # Extract and validate domain
    domain = extract_domain(parsed_url)
    validate_domain(domain)
    
    # Validate endpoint
    validate_endpoint(endpoint_name)
    
    # Build the base URL with endpoint
    base_url = build_base_url(parsed_url, endpoint_name)
    
    # Add query parameters if provided
    final_url = append_query_params(base_url, params_dict)
    
    return final_url
