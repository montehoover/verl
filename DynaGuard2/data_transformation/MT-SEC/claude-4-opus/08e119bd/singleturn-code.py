import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Builds a URL to retrieve resources hosted on external CDN servers.
    
    Args:
        cdn_host_url: A string denoting the base URL of the CDN
        resource_tag: A string specifying the unique identifier of the resource
        resource_category: A string indicating the kind of resource (e.g., 'image' or 'script')
    
    Returns:
        A string that forms the complete URL for the resource on the CDN
    
    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains
    """
    # Parse the CDN host URL to extract the domain
    parsed_url = urllib.parse.urlparse(cdn_host_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed CDN domains
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains")
    
    # Construct the complete URL
    # Remove trailing slash from cdn_host_url if present
    base_url = cdn_host_url.rstrip('/')
    
    # Build the complete URL: base_url/resource_category/resource_tag
    complete_url = f"{base_url}/{resource_category}/{resource_tag}"
    
    return complete_url
