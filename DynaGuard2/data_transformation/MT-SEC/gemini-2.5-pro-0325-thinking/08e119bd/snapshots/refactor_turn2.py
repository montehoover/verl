import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _is_valid_cdn_domain(cdn_host_url: str) -> bool:
    """
    Checks if the CDN host domain is in the allowed list.

    Args:
        cdn_host_url: The base URL of the CDN.

    Returns:
        True if the domain is allowed, False otherwise.
    """
    parsed_cdn_url = urllib.parse.urlparse(cdn_host_url)
    return parsed_cdn_url.hostname in ALLOWED_CDN_DOMAINS

def _build_resource_path(resource_category: str, resource_tag: str) -> str:
    """
    Constructs the resource path from category and tag.

    Args:
        resource_category: The kind of resource (e.g., 'image' or 'script').
        resource_tag: The unique identifier of the resource.

    Returns:
        The constructed resource path string.
    """
    # Construct the path, ensuring no double slashes from parts
    path_parts = [resource_category.strip('/'), resource_tag.strip('/')]
    return "/".join(part for part in path_parts if part)


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
    if not _is_valid_cdn_domain(cdn_host_url):
        # We parse the hostname again here for the error message to keep the
        # _is_valid_cdn_domain helper function simple (returning only boolean).
        hostname = urllib.parse.urlparse(cdn_host_url).hostname
        raise ValueError(
            f"CDN host domain '{hostname}' is not in the allowed list of CDN domains."
        )

    # Ensure cdn_host_url ends with a slash if it doesn't have one
    if not cdn_host_url.endswith('/'):
        cdn_host_url += '/'

    resource_path = _build_resource_path(resource_category, resource_tag)
    
    final_url = urllib.parse.urljoin(cdn_host_url, resource_path)
    
    return final_url
