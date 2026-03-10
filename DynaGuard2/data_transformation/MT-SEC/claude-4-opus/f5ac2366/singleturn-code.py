import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def construct_external_resource_url(cdn_base_url, resource_id, resource_type):
    """
    Constructs a URL for accessing external resources from a CDN.
    
    Args:
        cdn_base_url: A string representing the base URL of the CDN.
        resource_id: A string representing the identifier of the resource to be fetched.
        resource_type: A string indicating the type of resource (e.g., 'image', 'script').
    
    Returns:
        A string that is the complete URL for accessing the external resource.
    
    Raises:
        ValueError: If the resulting URL is not pointing to an allowed CDN domain.
    """
    # Construct the complete URL
    complete_url = f"{cdn_base_url}/{resource_type}/{resource_id}"
    
    # Parse the URL to extract the domain
    parsed_url = urllib.parse.urlparse(complete_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains")
    
    return complete_url
