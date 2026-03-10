from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    """
    Ensures the domain of an image URL belongs to a list of trusted image hosting services.

    Args:
        img_url: The URL of the image to resolve.

    Returns:
        The verified image URL if the domain is approved.

    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain
                    or if the scheme is unrecognized.
    """
    parsed_url = urlparse(img_url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unrecognized scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )

    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted image host: {parsed_url.hostname}. "
            f"Trusted hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}"
        )

    return img_url
