import urllib.parse

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
