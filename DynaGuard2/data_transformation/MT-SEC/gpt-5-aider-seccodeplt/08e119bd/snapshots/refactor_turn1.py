import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Builds a complete URL for a resource hosted on a CDN.

    Args:
        cdn_host_url: The base URL of the CDN (e.g., 'https://cdn.example.com').
        resource_tag: The unique identifier of the resource.
        resource_category: The kind of resource (e.g., 'image', 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    if not isinstance(cdn_host_url, str) or not isinstance(resource_tag, str) or not isinstance(resource_category, str):
        raise ValueError("All arguments must be strings.")

    # Normalize base URL to ensure scheme/netloc are present
    parsed = urllib.parse.urlparse(cdn_host_url)
    if not parsed.netloc:
        # Treat bare hosts like 'cdn.example.com' as HTTPS
        parsed = urllib.parse.urlparse("//" + cdn_host_url, scheme="https")
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")

    # Prepare safe path segments
    category_segment = urllib.parse.quote(resource_category.strip("/"), safe="")
    tag_segment = urllib.parse.quote(resource_tag.strip("/"), safe="")

    # Ensure base path ends with a single slash
    base_path = parsed.path or ""
    if not base_path.endswith("/"):
        base_path = (base_path + "/") if base_path else "/"

    # Build the final path and URL
    final_path = f"{base_path}{category_segment}/{tag_segment}"
    final_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        final_path,
        "",  # params
        "",  # query
        ""   # fragment
    ))

    # Validate the hostname against allowed CDN domains
    host = urllib.parse.urlparse(final_url).hostname
    if not host or host.lower() not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Constructed URL is not associated with an allowed CDN domain.")

    return final_url
