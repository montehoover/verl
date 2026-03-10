import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

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

def create_resource_url(base_cdn_url, res_id, res_type):
    # Parse the base CDN URL
    parsed_url = urllib.parse.urlparse(base_cdn_url)
    
    # Validate the domain
    if parsed_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain {parsed_url.netloc} is not in the allowed CDN domains")
    
    # Construct the path
    path = f"/{res_type}/{res_id}"
    
    # Build the complete URL
    complete_url = urllib.parse.urljoin(base_cdn_url, path)
    
    return complete_url
