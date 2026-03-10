from urllib.parse import urlparse, urlencode
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, netloc, path, params, query, fragment)
        
    Raises:
        ValueError: If the URL is invalid or missing required components
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Validate that the URL has at least a scheme and netloc
    if not parsed.scheme:
        raise ValueError("URL must have a valid scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must have a valid network location (domain)")
    
    # Return the parsed components as a dictionary
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }


def generate_query_string(params):
    """
    Generate a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): A dictionary of query parameters
        
    Returns:
        str: A URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)


def build_api_redirect_url(api_base_url, endpoint, query_params):
    """
    Construct a complete API redirect URL with validation.
    
    Args:
        api_base_url (str): The base URL of the API
        endpoint (str): The API endpoint
        query_params (dict): Dictionary of query parameters
        
    Returns:
        str: A complete redirect URL
        
    Raises:
        ValueError: If the base domain is not approved or endpoint is not allowed
    """
    # Parse and validate the base URL
    parsed_base = parse_and_validate_url(api_base_url)
    
    # Extract domain from netloc (remove port if present)
    domain = parsed_base['netloc'].split(':')[0]
    
    # Validate domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Validate endpoint is allowed
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in the allowed endpoints list")
    
    # Build the complete URL
    # Remove trailing slash from base URL if present
    base_url = api_base_url.rstrip('/')
    
    # Ensure endpoint starts with /
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    
    # Generate query string
    query_string = generate_query_string(query_params)
    
    # Construct full URL
    full_url = base_url + endpoint
    if query_string:
        full_url += '?' + query_string
    
    return full_url
