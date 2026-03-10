from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def validate_image_url(img_url: str) -> str:
    """
    Validate that the provided image URL uses an allowed scheme and that its domain
    is in the trusted image hosting providers list. Return the full URL if valid.

    Args:
        img_url: The input image URL string.

    Returns:
        The resolved valid URL string.

    Raises:
        ValueError: If the URL scheme is unsupported, the URL is malformed, or
                    the hostname is not in the trusted list.
    """
    if not isinstance(img_url, str):
        raise ValueError("img_url must be a string")

    url = img_url.strip()
    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: '{parsed.scheme}'")

    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL: missing hostname")

    # Only allow exact matches to trusted hosts
    if host not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted image host: '{host}'")

    # Return the full URL as provided (already parsed/validated)
    return parsed.geturl()
