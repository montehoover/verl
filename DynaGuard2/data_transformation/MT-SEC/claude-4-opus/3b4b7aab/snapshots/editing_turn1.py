def build_url_with_query(base_url, path, query_param):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Remove leading slash from path if present
    if path.startswith('/'):
        path = path[1:]
    
    # Combine base_url and path
    full_url = f"{base_url}/{path}"
    
    # Add query parameter
    if query_param:
        full_url = f"{full_url}?{query_param}"
    
    return full_url
