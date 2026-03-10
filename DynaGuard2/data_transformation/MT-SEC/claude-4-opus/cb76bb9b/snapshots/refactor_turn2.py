import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def construct_resource_url(base_cdn_url, res_id, res_type):
    """Constructs a resource URL from base CDN URL, resource ID, and resource type."""
    return urllib.parse.urljoin(base_cdn_url, f"{res_type}/{res_id}")

def validate_cdn_domain(url, allowed_domains):
    """Validates that the URL's domain is in the allowed CDN domains set."""
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in allowed_domains:
        raise ValueError(f"URL domain '{domain}' is not in the allowed CDN domains")
    
    return True

def create_resource_url(base_cdn_url, res_id, res_type):
    # Construct the complete URL
    url = construct_resource_url(base_cdn_url, res_id, res_type)
    
    # Validate the domain
    validate_cdn_domain(url, ALLOWED_CDN_DOMAINS)
    
    return url
