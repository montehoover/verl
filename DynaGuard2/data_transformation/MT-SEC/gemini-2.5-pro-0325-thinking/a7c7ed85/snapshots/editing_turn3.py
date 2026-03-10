from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates the image URL based on trusted domains and supported schemes.

    Args:
        img_url: The input image URL.

    Returns:
        The full URL if it comes from a trusted domain and has a supported scheme.

    Raises:
        ValueError: If the URL scheme is unsupported or the domain is not trusted.
    """
    parsed_url = urlparse(img_url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Must be one of {ALLOWED_SCHEMES}.")

    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Domain not trusted: {parsed_url.netloc}. Must be one of {TRUSTED_IMAGE_HOSTS}.")

    return img_url
