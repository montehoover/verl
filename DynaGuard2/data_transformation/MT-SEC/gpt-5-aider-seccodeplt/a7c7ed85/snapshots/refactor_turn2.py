from urllib.parse import urlparse, ParseResult

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _ensure_string(value, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _clean_url(url: str) -> str:
    return url.strip()


def _parse_url(url: str) -> ParseResult:
    return urlparse(url)


def _get_scheme(parsed: ParseResult) -> str:
    return (parsed.scheme or "").lower()


def _is_allowed_scheme(scheme: str) -> bool:
    return scheme in ALLOWED_SCHEMES


def _validate_scheme(scheme: str) -> None:
    if not _is_allowed_scheme(scheme):
        raise ValueError(f"Unsupported URL scheme: '{scheme}'")


def _get_hostname(parsed: ParseResult) -> str:
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL: missing hostname")
    return host


def _is_trusted_host(host: str) -> bool:
    # Exact host match only (no subdomain wildcards)
    return host in TRUSTED_IMAGE_HOSTS


def _validate_host(host: str) -> None:
    if not _is_trusted_host(host):
        raise ValueError(f"Untrusted image host: '{host}'")


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
    _ensure_string(img_url, "img_url")

    url = _clean_url(img_url)
    parsed = _parse_url(url)

    scheme = _get_scheme(parsed)
    _validate_scheme(scheme)

    host = _get_hostname(parsed)
    _validate_host(host)

    return parsed.geturl()
