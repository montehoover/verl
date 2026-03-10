import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def construct_external_resource_url(cdn_base_url, resource_id, resource_type):
    # Construct the complete URL
    if not cdn_base_url.endswith('/'):
        cdn_base_url += '/'
    
    # Build the resource path based on type
    resource_path = f"{resource_type}/{resource_id}"
    
    # Combine base URL with resource path
    complete_url = urllib.parse.urljoin(cdn_base_url, resource_path)
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(complete_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"URL domain '{domain}' is not in the list of allowed CDN domains")
    
    return complete_url
