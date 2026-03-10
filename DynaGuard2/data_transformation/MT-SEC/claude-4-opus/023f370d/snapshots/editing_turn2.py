import urllib.parse

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
