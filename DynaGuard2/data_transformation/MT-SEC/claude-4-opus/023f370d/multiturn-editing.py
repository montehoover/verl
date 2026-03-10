import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def build_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path

def build_url_with_query(base_url, path, query_params=None):
    # Validate base URL
    parsed_url = urllib.parse.urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Invalid base URL")
    
    # Build the base URL with path
    full_url = build_url(base_url, path)
    
    # Add query parameters if provided
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        full_url = f"{full_url}?{query_string}"
    
    return full_url

def construct_api_redirect(base_api_url, api_endpoint, query_options=None):
    # Parse and validate the base URL
    parsed_url = urllib.parse.urlparse(base_api_url)
    
    # Check if URL has scheme and netloc
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Invalid base API URL format")
    
    # Extract domain from netloc (remove port if present)
    domain = parsed_url.netloc.split(':')[0]
    
    # Validate domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Validate endpoint is allowed
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not in the allowed endpoints list")
    
    # Construct the full URL
    if parsed_url.path and parsed_url.path != '/':
        # If base URL has a path, combine it with endpoint
        path = urllib.parse.urljoin(parsed_url.path, api_endpoint.lstrip('/'))
    else:
        path = api_endpoint
    
    # Build URL components
    url_components = list(urllib.parse.urlparse(base_api_url))
    url_components[2] = path  # Set path
    
    # Add query parameters if provided
    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        url_components[4] = query_string  # Set query
    
    # Construct and return the final URL
    return urllib.parse.urlunparse(url_components)
