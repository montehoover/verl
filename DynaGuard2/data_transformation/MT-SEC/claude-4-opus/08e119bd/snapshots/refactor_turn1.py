import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

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
    # Parse the CDN host URL to extract the domain
    parsed_url = urllib.parse.urlparse(cdn_host_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"CDN domain '{domain}' is not in the allowed list of domains")
    
    # Ensure the base URL ends with a slash
    if not cdn_host_url.endswith('/'):
        cdn_host_url += '/'
    
    # Construct the complete URL
    complete_url = urllib.parse.urljoin(cdn_host_url, f"{resource_category}/{resource_tag}")
    
    return complete_url
