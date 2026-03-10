from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def resolve_image_url(image_url: str) -> str:
    """
    Resolves an image URL after verifying that it belongs to a trusted image hosting service.

    Args:
        image_url: The image URL to be resolved.

    Returns:
        The resolved URL if it belongs to a trusted domain.

    Raises:
        ValueError: If the image URL's scheme is not allowed or its domain is not in the trusted list.
    """
    parsed_url = urlparse(image_url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )

    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted image host: {parsed_url.hostname}. Trusted hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}"
        )

    return image_url
