import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Builds a URL to retrieve certain types of resources hosted on external servers.

    Args:
        cdn_host_url: The base URL of the CDN.
        resource_tag: The unique identifier of the resource.
        resource_category: The kind of resource (e.g., 'image' or 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    parsed_cdn_url = urllib.parse.urlparse(cdn_host_url)
    if parsed_cdn_url.hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"CDN host domain '{parsed_cdn_url.hostname}' is not in the allowed list of CDN domains."
        )

    # Ensure cdn_host_url ends with a slash if it doesn't have one
    if not cdn_host_url.endswith('/'):
        cdn_host_url += '/'

    # Construct the path, ensuring no double slashes if resource_category starts with one
    path_parts = [resource_category.strip('/'), resource_tag.strip('/')]
    resource_path = "/".join(part for part in path_parts if part)
    
    final_url = urllib.parse.urljoin(cdn_host_url, resource_path)
    
    return final_url
