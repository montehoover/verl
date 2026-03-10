import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def construct_internal_resource_url(base_url, resource_id):
    return base_url + resource_id

def construct_categorized_resource_url(base_url, resource_id, resource_type):
    valid_types = ['document', 'video', 'image', 'audio', 'data']
    if resource_type not in valid_types:
        raise ValueError(f"Invalid resource type. Must be one of: {', '.join(valid_types)}")
    return base_url + resource_type + '/' + resource_id

def construct_external_resource_url(cdn_base_url, resource_id, resource_type):
    # Parse the base URL
    parsed_url = urllib.parse.urlparse(cdn_base_url)
    
    # Validate the domain
    if parsed_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Invalid CDN domain. Must be one of: {', '.join(ALLOWED_CDN_DOMAINS)}")
    
    # Construct the full URL
    path = f"/{resource_type}/{resource_id}"
    full_url = urllib.parse.urljoin(cdn_base_url, path)
    
    # Parse the full URL to ensure it's still pointing to an allowed domain
    parsed_full_url = urllib.parse.urlparse(full_url)
    if parsed_full_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Resulting URL domain is not allowed: {parsed_full_url.netloc}")
    
    return full_url
