def generate_external_resource_link(cdn_host_url: str, resource_tag: str) -> str:
    """
    Constructs a URL to access resources hosted on a CDN.

    Args:
        cdn_host_url: The base CDN's URL.
        resource_tag: A unique identifier for the resource.

    Returns:
        The complete URL as a string.
    """
    return f"{cdn_host_url}{resource_tag}"
