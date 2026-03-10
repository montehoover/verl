from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    """
    Verifies an image URL by ensuring it uses a recognized scheme ('http' or 'https')
    and that its domain is one of the trusted image hosting services.

    Args:
        img_url: The input image URL string.

    Returns:
        The verified image URL string if the domain is approved and scheme is recognized.

    Raises:
        ValueError: If the URL scheme is unrecognized or the domain is not trusted.
    """
    parsed_url = urlparse(img_url)

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. Must be one of {ALLOWED_SCHEMES}."
        )

    if not parsed_url.netloc or parsed_url.netloc.lower() not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Domain '{parsed_url.netloc}' is not a trusted image host. "
            f"Must be one of {TRUSTED_IMAGE_HOSTS}."
        )

    return img_url
