import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def _build_resource_url(cdn_base_url, resource_id, resource_type):
    """Construct the complete URL from base URL and resource information."""
    if not cdn_base_url.endswith('/'):
        cdn_base_url += '/'
    
    resource_path = f"{resource_type}/{resource_id}"
    return urllib.parse.urljoin(cdn_base_url, resource_path)

def _validate_cdn_domain(url):
    """Validate that the URL points to an allowed CDN domain."""
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"URL domain '{domain}' is not in the list of allowed CDN domains")

def construct_external_resource_url(cdn_base_url, resource_id, resource_type):
    # Construct the complete URL
    complete_url = _build_resource_url(cdn_base_url, resource_id, resource_type)
    
    # Validate the domain
    _validate_cdn_domain(complete_url)
    
    return complete_url
