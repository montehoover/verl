from urllib.parse import urlencode, urlparse
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path

def construct_url_with_params(base_url, path, query_params=None):
    # Ensure base URL starts with https://
    if not base_url.startswith('https://'):
        if base_url.startswith('http://'):
            base_url = base_url.replace('http://', 'https://', 1)
        else:
            base_url = 'https://' + base_url
    
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    # Construct the URL
    url = base_url + path
    
    # Add query parameters if provided
    if query_params:
        query_string = urlencode(query_params)
        url = url + '?' + query_string
    
    return url

def create_api_redirect_link(base_url, path, params=None):
    # Parse the base URL to extract domain
    parsed_url = urlparse(base_url)
    
    # Extract domain (netloc) from parsed URL
    domain = parsed_url.netloc
    if not domain:
        # If no netloc, try parsing as if base_url is just the domain
        parsed_url = urlparse('https://' + base_url)
        domain = parsed_url.netloc
    
    # Validate domain against approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Validate endpoint path
    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not in the allowed endpoints list")
    
    # Ensure base URL has https scheme
    if parsed_url.scheme and parsed_url.scheme != 'https':
        raise ValueError("Base URL must use HTTPS protocol")
    
    # Reconstruct base URL with https if needed
    if not parsed_url.scheme:
        base_url = 'https://' + base_url
    
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Construct the full URL
    full_url = base_url + path
    
    # Add query parameters if provided
    if params:
        query_string = urlencode(params)
        full_url = full_url + '?' + query_string
    
    return full_url
