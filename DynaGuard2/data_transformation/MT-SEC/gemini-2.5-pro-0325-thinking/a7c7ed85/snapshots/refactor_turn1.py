from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates an image URL, confirming its domain is part of a trusted list
    and its scheme is supported.

    Args:
        img_url: The input image URL.

    Returns:
        The resolved valid URL if it belongs to the list of trusted domains
        and supported schemes.

    Raises:
        ValueError: If the domain is not in the trusted list, the scheme is
                    unsupported, or the URL is malformed.
    """
    try:
        parsed_url = urlparse(img_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {img_url}") from e

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL, missing scheme or domain: {img_url}")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )

    if parsed_url.netloc.lower() not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted image host: {parsed_url.netloc}. "
            f"Allowed hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}"
        )

    return img_url
