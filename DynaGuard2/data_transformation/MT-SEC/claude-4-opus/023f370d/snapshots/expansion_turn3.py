from urllib.parse import urlparse, urlencode
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_and_parse_url(url):
    """
    Validates and parses a URL string.
    
    Args:
        url (str): The URL string to validate and parse
        
    Returns:
        dict: A dictionary containing 'scheme', 'domain', and 'path'
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme is present
    if not parsed.scheme:
        raise ValueError("Invalid URL: missing scheme (e.g., http://, https://)")
    
    # Check if netloc (domain) is present
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }


def build_query_string(params):
    """
    Builds a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of query parameters
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)


def construct_api_redirect(base_api_url, api_endpoint, query_options):
    """
    Constructs a complete API redirect URL with validation.
    
    Args:
        base_api_url (str): The base API URL
        api_endpoint (str): The API endpoint path
        query_options (dict): Dictionary of query parameters
        
    Returns:
        str: Fully constructed redirect URL
        
    Raises:
        ValueError: If base domain is not approved or endpoint is not allowed
    """
    # Parse and validate the base URL
    parsed_url = validate_and_parse_url(base_api_url)
    
    # Check if domain is approved
    if parsed_url['domain'] not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_url['domain']}' is not in the approved API domains list")
    
    # Normalize endpoint (ensure it starts with /)
    if not api_endpoint.startswith('/'):
        api_endpoint = '/' + api_endpoint
    
    # Check if endpoint is allowed
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not in the allowed endpoints list")
    
    # Build the query string
    query_string = build_query_string(query_options)
    
    # Construct the full URL
    full_url = f"{parsed_url['scheme']}://{parsed_url['domain']}{api_endpoint}"
    
    if query_string:
        full_url += f"?{query_string}"
    
    return full_url
