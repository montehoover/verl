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
        dict: A dictionary containing the URL components (scheme, domain, path)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme is present and valid
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http:// or https://)")
    
    # Check if domain/netloc is present
    if not parsed.netloc:
        raise ValueError("URL must include a domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }


def build_query_string(params):
    """
    Builds a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)


def create_api_redirect_url(base_url_api, api_endpoint, opts):
    """
    Creates a fully-formed API redirect URL with validation.
    
    Args:
        base_url_api (str): The base API URL
        api_endpoint (str): The API endpoint path
        opts (dict): Dictionary of query parameters
        
    Returns:
        str: Complete redirect URL
        
    Raises:
        ValueError: If base domain is not approved or endpoint is not allowed
    """
    # Parse the base URL to extract domain
    parsed = urllib.parse.urlparse(base_url_api)
    domain = parsed.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved list")
    
    # Validate endpoint
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not allowed")
    
    # Build the full URL
    if base_url_api.endswith('/'):
        base_url_api = base_url_api.rstrip('/')
    
    if not api_endpoint.startswith('/'):
        api_endpoint = '/' + api_endpoint
    
    full_url = base_url_api + api_endpoint
    
    # Add query parameters if provided
    if opts:
        query_string = build_query_string(opts)
        full_url = full_url + '?' + query_string
    
    return full_url
