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
    # Ensure base_url starts with https://
    if not base_url.startswith('https://'):
        base_url = 'https://' + base_url
    
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    # Construct the URL without query params
    url = base_url + path
    
    # Add query parameters if provided
    if query_params:
        query_string = urlencode(query_params)
        url = url + '?' + query_string
    
    return url

def generate_api_redirect(base_url, api_path, params=None):
    # Parse the base URL to extract domain
    parsed = urlparse(base_url)
    
    # If no scheme is provided, assume https
    if not parsed.scheme:
        base_url = 'https://' + base_url
        parsed = urlparse(base_url)
    
    # Validate the domain
    domain = parsed.netloc
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains")
    
    # Validate the endpoint
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not in the allowed endpoints")
    
    # Build the full URL
    full_url = urllib.parse.urljoin(base_url, api_path)
    
    # Add query parameters if provided
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url = full_url + '?' + query_string
    
    return full_url
