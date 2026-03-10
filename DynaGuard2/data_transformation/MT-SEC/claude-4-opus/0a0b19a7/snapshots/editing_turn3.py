import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path

def construct_url_with_params(base_url, path, query_params=None):
    # Validate base URL has a valid domain
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base URL: must include scheme and domain")
    
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    # Construct base URL with path
    url = base_url + path
    
    # Add query parameters if provided
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        url = url + '?' + query_string
    
    return url

def build_api_redirect_url(api_base_url, endpoint, query_params=None):
    # Parse the base URL
    parsed_url = urllib.parse.urlparse(api_base_url)
    
    # Validate domain
    if parsed_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_url.netloc}' is not in approved API domains")
    
    # Validate endpoint
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not in allowed endpoints")
    
    # Build the URL components
    scheme = parsed_url.scheme if parsed_url.scheme else 'https'
    netloc = parsed_url.netloc
    path = endpoint
    
    # Build query string if parameters provided
    query = ''
    if query_params:
        query = urllib.parse.urlencode(query_params)
    
    # Construct the final URL
    url_components = urllib.parse.ParseResult(
        scheme=scheme,
        netloc=netloc,
        path=path,
        params='',
        query=query,
        fragment=''
    )
    
    return urllib.parse.urlunparse(url_components)
