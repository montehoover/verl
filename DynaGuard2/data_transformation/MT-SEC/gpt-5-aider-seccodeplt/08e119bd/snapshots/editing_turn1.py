def generate_external_resource_link(cdn_host_url, resource_tag):
    """
    Concatenate the CDN base URL and resource tag to form the full resource URL.

    Args:
        cdn_host_url (str): Base CDN URL.
        resource_tag (str): Unique resource identifier.

    Returns:
        str: The full URL to the resource.
    """
    return f"{cdn_host_url}{resource_tag}"
