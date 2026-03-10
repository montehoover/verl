import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Constructs a URL to retrieve a resource from a CDN.

    The URL is built using urllib.parse and validated against a predefined
    set of allowed CDN domains.

    Args:
        base_cdn_url: The base address of the CDN (e.g., "https://cdn.example.com").
        res_id: The unique identifier of the resource (e.g., "12345").
        res_type: The kind of resource (e.g., "image", "script").

    Returns:
        The full URL as a string.

    Raises:
        ValueError: If the base_cdn_url is not associated with any allowed CDN domains,
                    or if base_cdn_url is malformed.
    """
    parsed_base_url = urllib.parse.urlparse(base_cdn_url)

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_cdn_url: '{base_cdn_url}'. It must include a scheme and domain.")

    if parsed_base_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_base_url.netloc}' is not an allowed CDN domain. "
            f"Allowed domains are: {', '.join(ALLOWED_CDN_DOMAINS)}"
        )

    # Ensure base_cdn_url ends with a slash for proper joining
    if not base_cdn_url.endswith('/'):
        base_cdn_url += '/'

    # Construct the path part: res_type/res_id
    # Ensure res_type and res_id are URL-safe, though not explicitly requested, it's good practice.
    # For this specific request, we'll just join them.
    # We also need to ensure no leading slash on the path components if base_cdn_url already has one.
    
    path_segment = f"{res_type}/{res_id}"
    
    # urljoin handles slashes correctly between the base and the path.
    # If base_cdn_url is "http://domain.com/" and path_segment is "image/123",
    # result is "http://domain.com/image/123".
    # If base_cdn_url is "http://domain.com" (no trailing slash) and path_segment is "image/123",
    # urljoin might replace the last component of base if it thinks it's a file.
    # So, ensuring base_cdn_url ends with '/' is safer.
    
    full_url = urllib.parse.urljoin(base_cdn_url, path_segment)
    
    return full_url
