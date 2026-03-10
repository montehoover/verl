import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def create_resource_url(base_cdn_url, res_id, res_type):
    """
    Build a URL to retrieve resources from a CDN.
    
    Args:
        base_cdn_url: a string denoting the base URL of the CDN
        res_id: a string specifying the unique identifier of the resource
        res_type: a string indicating the kind of resource (e.g., 'image' or 'script')
    
    Returns:
        A string that forms the complete URL for the resource on the CDN
    
    Raises:
        ValueError: if the constructed URL is not associated with any allowed CDN domains
    """
    # Parse the base URL to extract the domain
    parsed_url = urllib.parse.urlparse(base_cdn_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains")
    
    # Construct the complete URL
    # Remove trailing slash from base_cdn_url if present
    base_cdn_url = base_cdn_url.rstrip('/')
    
    # Build the complete URL with the pattern: base_url/res_type/res_id
    complete_url = f"{base_cdn_url}/{res_type}/{res_id}"
    
    return complete_url
