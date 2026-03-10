import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def extract_domain(url):
    """Extract the domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc

def is_valid_cdn_domain(domain):
    """Check if a domain is in the allowed CDN domains list."""
    return domain in ALLOWED_CDN_DOMAINS

def build_resource_path(resource_category, resource_tag):
    """Construct the resource path from category and tag."""
    return f"{resource_category}/{resource_tag}"

def normalize_base_url(url):
    """Ensure the base URL ends with a slash."""
    if not url.endswith('/'):
        return url + '/'
    return url

def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Build a URL to retrieve certain types of resources hosted on external servers.
    
    Args:
        cdn_host_url: a string denoting the base URL of the CDN
        resource_tag: a string specifying the unique identifier of the resource
        resource_category: a string indicating the kind of resource (e.g., 'image' or 'script')
    
    Returns:
        A string that forms the complete URL for the resource on the CDN
    
    Raises:
        ValueError: if the constructed URL is not associated with any allowed CDN domains
    """
    # Extract and validate domain
    domain = extract_domain(cdn_host_url)
    if not is_valid_cdn_domain(domain):
        raise ValueError(f"CDN domain '{domain}' is not in the allowed list of domains")
    
    # Normalize the base URL
    normalized_url = normalize_base_url(cdn_host_url)
    
    # Build the resource path
    resource_path = build_resource_path(resource_category, resource_tag)
    
    # Construct the complete URL
    complete_url = urllib.parse.urljoin(normalized_url, resource_path)
    
    return complete_url
