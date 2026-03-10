import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Construct a URL for accessing an external resource from a CDN.

    Args:
        cdn_base_url: Base URL of the CDN (e.g., "https://cdn.example.com").
        resource_id: Identifier of the resource to be fetched.
        resource_type: Type of resource (e.g., "image", "script").

    Returns:
        Full URL string to the external resource.

    Raises:
        ValueError: If the resulting URL does not point to an allowed CDN domain.
    """
    # Safely encode path segments and prevent leading/trailing slashes from affecting the path.
    encoded_type = urllib.parse.quote(str(resource_type).strip('/'), safe='-._~')
    encoded_id = urllib.parse.quote(str(resource_id).strip('/'), safe='-._~')

    # Ensure base URL is treated as a directory for urljoin semantics.
    base = (str(cdn_base_url or '').rstrip('/') + '/')

    # Construct the relative path and join with the base URL.
    relative_path = f"{encoded_type}/{encoded_id}"
    full_url = urllib.parse.urljoin(base, relative_path)

    # Validate the resulting URL's domain.
    parsed = urllib.parse.urlparse(full_url)
    if parsed.hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Resulting URL domain is not allowed.")

    return full_url
