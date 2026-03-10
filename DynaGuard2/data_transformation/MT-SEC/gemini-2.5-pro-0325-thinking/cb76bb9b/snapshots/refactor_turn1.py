import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Builds a URL to retrieve certain types of resources hosted on external servers,
    using the base URL of a CDN and a user-defined resource identifier.

    Args:
        base_cdn_url: A string denoting the base URL of the CDN.
        res_id: A string specifying the unique identifier of the resource.
        res_type: A string indicating the kind of resource (e.g., 'image' or 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    parsed_url = urllib.parse.urlparse(base_cdn_url)
    domain = parsed_url.netloc
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not an allowed CDN domain. "
            f"Allowed domains are: {', '.join(ALLOWED_CDN_DOMAINS)}"
        )

    # Ensure base_cdn_url ends with a slash for proper joining
    if not base_cdn_url.endswith('/'):
        base_cdn_url += '/'

    # Construct the path segment: res_type/res_id
    resource_path = f"{res_type}/{res_id}"

    # Join the base URL with the resource path
    full_url = urllib.parse.urljoin(base_cdn_url, resource_path)

    return full_url
