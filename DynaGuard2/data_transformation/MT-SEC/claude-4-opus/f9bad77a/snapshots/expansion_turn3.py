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
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Validate that we have at least a scheme and netloc for absolute URLs
    # or at least a path for relative URLs
    if not parsed.scheme and not parsed.netloc and not parsed.path:
        raise ValueError("Invalid URL: must have a scheme, netloc, or path")
    
    # If scheme is present, netloc should also be present for valid absolute URLs
    if parsed.scheme and not parsed.netloc:
        raise ValueError("Invalid URL: scheme present but no netloc")
    
    # Return the components as a dictionary
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

def construct_redirect_url(main_url, endpoint_name, params_dict):
    """
    Construct a complete API redirect URL with validation.
    
    Args:
        main_url (str): The base URL
        endpoint_name (str): The API endpoint name
        params_dict (dict): Dictionary of query parameters
        
    Returns:
        str: Full redirect URL
        
    Raises:
        ValueError: If domain is not approved or endpoint is not allowed
    """
    # Parse the main URL
    parsed = urlparse(main_url)
    
    # Extract and validate domain
    domain = parsed.netloc
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved list")
    
    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not allowed")
    
    # Construct the base URL with endpoint
    base_url = f"{parsed.scheme}://{parsed.netloc}{endpoint_name}"
    
    # Add query parameters if provided
    if params_dict:
        query_string = generate_query_string(params_dict)
        base_url = f"{base_url}?{query_string}"
    
    return base_url
