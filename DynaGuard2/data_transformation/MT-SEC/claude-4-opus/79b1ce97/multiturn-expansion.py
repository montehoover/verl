from urllib.parse import urlparse, urlencode
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def parse_and_validate_url(url):
    """
    Parse and validate a URL.
    
    Args:
        url (str): The URL to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, netloc, path, params, query, fragment)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if the URL has at least a scheme and netloc (for absolute URLs)
    # or at least a path (for relative URLs)
    if not parsed.scheme and not parsed.netloc and not parsed.path:
        raise ValueError("Invalid URL: missing required components")
    
    # If scheme is present, netloc should also be present for absolute URLs
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


def generate_query_string(params):
    """
    Generate a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)


def create_redirect_url(base_url_api, desired_endpoint, optional_params):
    """
    Create a complete API redirect URL with validation.
    
    Args:
        base_url_api (str): The base URL of the API
        desired_endpoint (str): The desired endpoint path
        optional_params (dict): Optional query parameters
        
    Returns:
        str: Complete redirect URL
        
    Raises:
        ValueError: If the base domain is not approved or endpoint is not allowed
    """
    # Parse the base URL
    parsed_base = urllib.parse.urlparse(base_url_api)
    
    # Extract domain (netloc)
    domain = parsed_base.netloc
    if not domain:
        raise ValueError("Invalid base URL: missing domain")
    
    # Check if domain is approved
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved list")
    
    # Check if endpoint is allowed
    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not allowed")
    
    # Construct the complete URL
    # Use the scheme from base_url_api or default to https
    scheme = parsed_base.scheme if parsed_base.scheme else 'https'
    
    # Build the URL
    url = f"{scheme}://{domain}{desired_endpoint}"
    
    # Add query parameters if provided
    if optional_params:
        query_string = generate_query_string(optional_params)
        url = f"{url}?{query_string}"
    
    return url
