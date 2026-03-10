import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Build a URL to retrieve a resource hosted on a CDN.

    Args:
        cdn_host_url: Base URL of the CDN (e.g., 'https://cdn.example.com').
        resource_tag: Unique identifier of the resource (e.g., 'abcd1234').
        resource_category: Kind of resource (e.g., 'image', 'script').

    Returns:
        A string forming the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the CDN host is not in the allowed CDN domains.
    """
    parsed = urllib.parse.urlparse(cdn_host_url)
    host = parsed.hostname

    # Validate that the host is an allowed CDN domain
    if not host or host not in ALLOWED_CDN_DOMAINS:
        raise ValueError("CDN host is not allowed")

    # Normalize and build the path:
    # - Preserve any existing base path on the CDN URL
    # - Append resource_category and resource_tag as path segments
    base_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    cat_seg = urllib.parse.quote(resource_category, safe="")
    tag_seg = urllib.parse.quote(resource_tag, safe="")
    path_segments = base_segments + [cat_seg, tag_seg]
    path = "/" + "/".join(path_segments)

    # Rebuild the full URL
    full_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        path,
        "",  # params
        "",  # query
        ""   # fragment
    ))

    return full_url
