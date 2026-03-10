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
        dict: A dictionary containing URL components (scheme, netloc, path, params, query, fragment)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Validate that we have at least a scheme and netloc for absolute URLs
    # or just a path for relative URLs
    if not parsed.scheme and not parsed.netloc and not parsed.path:
        raise ValueError("Invalid URL: missing required components")
    
    # If scheme is present, netloc should also be present for valid absolute URLs
    if parsed.scheme and not parsed.netloc:
        raise ValueError("Invalid URL: scheme present but missing netloc")
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }


def build_query_string(params):
    """
    Build a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of query parameters
        
    Returns:
        str: URL-encoded query string
    """
    return urlencode(params)


def create_api_redirect_link(base_url, path, params):
    """
    Generate a complete API redirect URL with validation.
    
    Args:
        base_url (str): The base URL of the API
        path (str): The API endpoint path
        params (dict): Dictionary of query parameters
        
    Returns:
        str: Fully constructed API URL
        
    Raises:
        ValueError: If the domain is not approved or endpoint is not allowed
    """
    # Parse the base URL to extract the domain
    parsed_base = urllib.parse.urlparse(base_url)
    domain = parsed_base.netloc
    
    # Validate domain
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved list")
    
    # Validate endpoint
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not allowed")
    
    # Construct the full URL
    # Ensure the base URL ends properly
    if base_url.endswith('/'):
        base_url = base_url.rstrip('/')
    
    # Ensure the path starts with /
    if not path.startswith('/'):
        path = '/' + path
    
    # Build the query string
    query_string = build_query_string(params) if params else ''
    
    # Construct the final URL
    full_url = base_url + path
    if query_string:
        full_url += '?' + query_string
    
    return full_url
