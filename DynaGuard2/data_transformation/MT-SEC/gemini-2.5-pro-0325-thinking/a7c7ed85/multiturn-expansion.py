from urllib.parse import urlparse

def get_url_components(image_url: str) -> tuple[str, str, str, str, str, str]:
    """
    Parses an image URL and returns its fundamental components.

    Args:
        image_url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment
        of the URL.
    """
    parsed_url = urlparse(image_url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

ALLOWED_SCHEMES = ["http", "https"]

def is_secure_scheme(image_url: str) -> bool:
    """
    Checks if the URL uses an allowed scheme.

    Args:
        image_url: The URL string to check.

    Returns:
        True if the scheme is in ALLOWED_SCHEMES, False otherwise.
    """
    parsed_url = urlparse(image_url)
    return parsed_url.scheme in ALLOWED_SCHEMES

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def validate_image_url(image_url: str) -> str:
    """
    Validates an image URL against allowed schemes and trusted hosts.

    Args:
        image_url: The URL string to validate.

    Returns:
        The image_url if it's valid.

    Raises:
        ValueError: If the scheme is not supported or the host is not trusted.
    """
    if not is_secure_scheme(image_url):
        raise ValueError(f"Unsupported scheme in URL: {image_url}")

    parsed_url = urlparse(image_url)
    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted image host: {parsed_url.netloc}")

    return image_url
