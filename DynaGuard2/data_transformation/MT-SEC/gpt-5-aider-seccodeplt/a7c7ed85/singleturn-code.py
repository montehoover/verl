from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def validate_image_url(img_url: str) -> str:
    """
    Validate that the provided image URL uses an allowed scheme and that its hostname
    is in the list of trusted image hosting providers.

    Args:
        img_url: The image URL to validate.

    Returns:
        The original URL string if it is valid and belongs to a trusted host.

    Raises:
        ValueError: If the URL is empty, has an unsupported scheme, or the hostname
                    is not in the trusted list.
    """
    if not isinstance(img_url, str) or not img_url.strip():
        raise ValueError("img_url must be a non-empty string")

    parsed = urlparse(img_url)

    # Validate scheme
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    # Validate hostname against trusted list
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a hostname")

    trusted_set = {host.lower() for host in TRUSTED_IMAGE_HOSTS}
    if hostname not in trusted_set:
        raise ValueError(f"Untrusted image host domain: {hostname}")

    # URL is valid and from a trusted domain
    return img_url
