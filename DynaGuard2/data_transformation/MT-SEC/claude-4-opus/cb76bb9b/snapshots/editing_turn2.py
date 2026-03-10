def build_url(base_url, path):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Add leading slash to path if not present
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path

def build_resource_url(base_url, path, res_type):
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Remove leading slash from path if present
    if path.startswith('/'):
        path = path[1:]
    
    # Construct the resource path
    resource_path = f"/{res_type}/{path}"
    
    return base_url + resource_path
