from urllib.parse import urlencode

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
