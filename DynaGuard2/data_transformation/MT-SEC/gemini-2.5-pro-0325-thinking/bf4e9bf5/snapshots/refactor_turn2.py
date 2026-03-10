from urllib.parse import urlparse, ParseResult

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_url(image_url: str) -> ParseResult:
    """Parses the given image URL."""
    return urlparse(image_url)


def _validate_scheme(parsed_url: ParseResult) -> None:
    """
    Validates the scheme of the parsed URL.

    Raises:
        ValueError: If the URL scheme is not in ALLOWED_SCHEMES.
    """
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )


def _validate_host(parsed_url: ParseResult) -> None:
    """
    Validates the hostname of the parsed URL.

    Raises:
        ValueError: If the URL hostname is not in TRUSTED_IMAGE_HOSTS.
    """
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted image host: {parsed_url.hostname}. "
            f"Trusted hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}"
        )


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
    parsed_url = _parse_url(image_url)

    _validate_scheme(parsed_url)
    _validate_host(parsed_url)

    return image_url
