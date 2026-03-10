from urllib.parse import urlparse

def extract_url_components(url: str) -> dict:
    """
    Extract scheme, domain, and path from a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with keys: 'scheme', 'domain', 'path'.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed = urlparse('//' + url)

    return {
        'scheme': parsed.scheme or '',
        'domain': parsed.hostname or '',
        'path': parsed.path or '',
    }
