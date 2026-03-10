from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def fetch_image_url(url: str) -> str:
    """
    Validate that the given image URL uses an allowed scheme and belongs to a trusted image host.

    Args:
        url: The URL of the image to resolve.

    Returns:
        The verified image URL if the domain is approved.

    Raises:
        ValueError: If the URL scheme is not allowed or the host is not trusted.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    parsed = urlparse(candidate)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unrecognized or disallowed URL scheme")

    host = parsed.hostname  # Lowercased by urlparse; excludes port/userinfo
    if not host:
        raise ValueError("URL must include a hostname")

    if host not in TRUSTED_IMAGE_HOSTS:
        raise ValueError("URL host is not in the list of trusted image hosts")

    return candidate
