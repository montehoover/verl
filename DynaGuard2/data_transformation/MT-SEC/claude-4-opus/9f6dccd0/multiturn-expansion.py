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
        dict: A dictionary containing the URL components (scheme, domain, path)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme and netloc (domain) are present for valid URL
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http://, https://)")
    
    if not parsed.netloc:
        raise ValueError("URL must include a domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }

def build_query_string(params):
    """
    Build a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of query parameters
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)

def generate_api_redirect(base_url, api_path, params):
    """
    Generate a complete API redirect URL with validation.
    
    Args:
        base_url (str): Base URL for the API
        api_path (str): API endpoint path
        params (dict): Query parameters to include
        
    Returns:
        str: Fully constructed redirect URL
        
    Raises:
        ValueError: If base domain is not approved or endpoint is not allowed
    """
    # Parse and validate the base URL
    parsed_url = parse_and_validate_url(base_url)
    
    # Check if domain is approved
    if parsed_url['domain'] not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain {parsed_url['domain']} is not in the approved list")
    
    # Check if endpoint is allowed
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint {api_path} is not allowed")
    
    # Build the full URL
    full_url = base_url.rstrip('/') + api_path
    
    # Add query parameters if any
    if params:
        query_string = build_query_string(params)
        full_url += '?' + query_string
    
    return full_url
