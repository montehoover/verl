import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Build a CDN resource URL of the form: <base_cdn_url>/<res_type>/<res_id>

    Args:
        base_cdn_url: Base URL of the CDN (e.g., "https://cdn.example.com")
        res_id: Unique identifier of the resource
        res_type: Type of resource (e.g., "image", "script")

    Returns:
        The complete URL for the resource on the CDN.

    Raises:
        ValueError: If the resulting URL is not associated with an allowed CDN domain.
    """
    # Parse base URL; tolerate missing scheme by assuming https
    parsed = urllib.parse.urlparse(base_cdn_url)
    if not parsed.scheme or not parsed.netloc:
        parsed = urllib.parse.urlparse('https://' + base_cdn_url.lstrip('/'))

    hostname = parsed.hostname.lower() if parsed.hostname else ''
    if hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Disallowed CDN domain: {hostname}")

    base_path = (parsed.path or '').rstrip('/')
    res_type_part = (res_type or '').strip('/')
    res_id_part = (res_id or '').strip('/')

    path_parts = [p for p in [base_path, res_type_part, res_id_part] if p]
    new_path = '/'.join(path_parts)
    if not new_path.startswith('/'):
        new_path = '/' + new_path

    url = urllib.parse.urlunparse((
        parsed.scheme or 'https',
        parsed.netloc,
        new_path,
        '', '', ''
    ))

    # Final safety check
    final_host = urllib.parse.urlparse(url).hostname
    if final_host is None or final_host.lower() not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Constructed URL not associated with an allowed CDN domain: {final_host}")

    return url
