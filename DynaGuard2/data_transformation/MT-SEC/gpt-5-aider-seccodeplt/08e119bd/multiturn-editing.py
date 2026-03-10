import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Construct the full resource URL in the form:
    {cdn_host_url}/{resource_category}/{resource_tag}
    Validates that the CDN host is in ALLOWED_CDN_DOMAINS.
    Raises ValueError if not allowed.

    Args:
        cdn_host_url (str): Base CDN URL.
        resource_tag (str): Unique resource identifier.
        resource_category (str): Category of the resource (e.g., images, scripts).

    Returns:
        str: The full URL to the resource.

    Raises:
        ValueError: If cdn_host_url's domain is not permitted.
    """
    raw = str(cdn_host_url).strip()

    # Parse URL; handle bare host by prepending a default scheme for parsing
    parts = urllib.parse.urlparse(raw)
    if not parts.netloc:
        parts = urllib.parse.urlparse(f"https://{raw.lstrip('/')}")

    hostname = parts.hostname
    if not hostname or hostname.lower() not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Unpermitted CDN domain")

    base_path = parts.path.rstrip('/')
    category = str(resource_category).strip('/')
    tag = str(resource_tag).lstrip('/')

    if base_path:
        new_path = f"{base_path}/{category}/{tag}"
    else:
        new_path = f"/{category}/{tag}"

    scheme = parts.scheme or 'https'
    return urllib.parse.urlunparse((scheme, parts.netloc, new_path, '', '', ''))
