from urllib.parse import urlparse, ParseResult

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_url(img_url: str) -> ParseResult:
    """Parses the URL and raises ValueError for malformed URLs."""
    try:
        parsed_url = urlparse(img_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {img_url}") from e

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL, missing scheme or domain: {img_url}")
    return parsed_url


def _validate_scheme(parsed_url: ParseResult, allowed_schemes: list[str]) -> None:
    """Validates the URL scheme against a list of allowed schemes."""
    if parsed_url.scheme.lower() not in allowed_schemes:
        raise ValueError(
            f"Unsupported scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(allowed_schemes)}"
        )


def _validate_host(parsed_url: ParseResult, trusted_hosts: list[str]) -> None:
    """Validates the URL host against a list of trusted hosts."""
    if parsed_url.netloc.lower() not in trusted_hosts:
        raise ValueError(
            f"Untrusted image host: {parsed_url.netloc}. "
            f"Allowed hosts are: {', '.join(trusted_hosts)}"
        )


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
    parsed_url = _parse_url(img_url)
    _validate_scheme(parsed_url, ALLOWED_SCHEMES)
    _validate_host(parsed_url, TRUSTED_IMAGE_HOSTS)

    return img_url
