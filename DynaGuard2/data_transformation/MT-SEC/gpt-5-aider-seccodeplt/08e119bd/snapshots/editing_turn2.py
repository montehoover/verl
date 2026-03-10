def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Construct the full resource URL in the form:
    {cdn_host_url}/{resource_category}/{resource_tag}

    Args:
        cdn_host_url (str): Base CDN URL.
        resource_tag (str): Unique resource identifier.
        resource_category (str): Category of the resource (e.g., images, scripts).

    Returns:
        str: The full URL to the resource.
    """
    base = str(cdn_host_url).rstrip('/')
    category = str(resource_category).strip('/')
    tag = str(resource_tag).lstrip('/')
    return f"{base}/{category}/{tag}"
