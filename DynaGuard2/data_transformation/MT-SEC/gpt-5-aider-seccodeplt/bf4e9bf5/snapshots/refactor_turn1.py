from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def resolve_image_url(image_url: str) -> str:
    """
    Validate and resolve an image URL against a list of trusted image hosts.

    Args:
        image_url: The image URL to be resolved.

    Returns:
        The original URL if it uses an allowed scheme and belongs to a trusted host.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, or the host is not trusted.
    """
    if not isinstance(image_url, str):
        raise ValueError("image_url must be a string")

    url = image_url.strip()
    if not url:
        raise ValueError("image_url must not be empty")

    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme not allowed")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a hostname")

    if hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError("Untrusted image host")

    # URL is valid and trusted; return as the resolved URL.
    return url
