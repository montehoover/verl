def construct_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path
