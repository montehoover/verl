def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Constructs a URL to access resources hosted on a CDN.

    Args:
        cdn_host_url: The base CDN's URL.
        resource_tag: A unique identifier for the resource.
        resource_category: The category of the resource (e.g., image, script).

    Returns:
        The complete URL as a string, with the resource_category as a directory.
    """
    # Ensure cdn_host_url does not end with a slash to prevent double slashes
    if cdn_host_url.endswith('/'):
        cdn_host_url = cdn_host_url[:-1]
    
    # Ensure resource_category and resource_tag do not start with a slash
    if resource_category.startswith('/'):
        resource_category = resource_category[1:]
    if resource_tag.startswith('/'):
        resource_tag = resource_tag[1:]

    return f"{cdn_host_url}/{resource_category}/{resource_tag}"
