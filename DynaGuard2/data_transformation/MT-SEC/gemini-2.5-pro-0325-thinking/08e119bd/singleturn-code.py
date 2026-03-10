import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Builds a URL to retrieve certain types of resources hosted on external servers,
    using the base URL of a CDN and a user-defined resource identifier.

    Args:
        cdn_host_url: A string denoting the base URL of the CDN.
        resource_tag: A string specifying the unique identifier of the resource.
        resource_category: A string indicating the kind of resource (e.g., 'image' or 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: if the constructed URL is not associated with any allowed CDN domains.
    """
    parsed_url = urllib.parse.urlparse(cdn_host_url)
    hostname = parsed_url.hostname

    if hostname is None or hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"CDN host '{hostname}' is not in the allowed domains.")

    # Ensure cdn_host_url ends with a slash for proper joining
    if not cdn_host_url.endswith('/'):
        cdn_host_url += '/'

    # Construct the path part
    resource_path = f"{resource_category}/{resource_tag}"

    # Join the base URL with the resource path
    full_url = urllib.parse.urljoin(cdn_host_url, resource_path)

    return full_url
