from urllib.parse import urlparse, ParseResult
import logging

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def parse_image_url(image_url: str) -> ParseResult:
    """
    Parse the provided image URL after basic input normalization and checks.

    Args:
        image_url: The raw image URL input.

    Returns:
        A urllib.parse.ParseResult representing the parsed URL.

    Raises:
        ValueError: If the input is not a string or is empty after trimming.
    """
    if not isinstance(image_url, str):
        raise ValueError("image_url must be a string")

    url = image_url.strip()
    if not url:
        raise ValueError("image_url must not be empty")

    return urlparse(url)


def is_allowed_scheme(scheme: str) -> bool:
    """
    Check whether the scheme is allowed.

    Args:
        scheme: URL scheme (e.g., 'http', 'https').

    Returns:
        True if the scheme is allowed; otherwise, False.
    """
    return scheme.lower() in ALLOWED_SCHEMES


def is_trusted_host(hostname: str) -> bool:
    """
    Check whether the hostname is in the trusted image hosts list.

    Args:
        hostname: The URL's hostname.

    Returns:
        True if the hostname is trusted; otherwise, False.
    """
    return hostname.lower() in TRUSTED_IMAGE_HOSTS


def validate_parsed_url(parsed: ParseResult) -> None:
    """
    Validate components of the parsed URL against the allowed schemes and trusted hosts.

    Args:
        parsed: The parsed URL object.

    Raises:
        ValueError: If the scheme is disallowed, hostname missing, or host untrusted.
    """
    scheme = (parsed.scheme or "").lower()
    if not is_allowed_scheme(scheme):
        raise ValueError("URL scheme not allowed")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a hostname")

    if not is_trusted_host(hostname):
        raise ValueError("Untrusted image host")


def resolve_image_url(image_url: str) -> str:
    """
    Validate and resolve an image URL against a list of trusted image hosts.
    Logs resolution attempts, successes, and errors to a file in the current directory.

    Args:
        image_url: The image URL to be resolved.

    Returns:
        The original URL if it uses an allowed scheme and belongs to a trusted host.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, or the host is not trusted.
    """
    logger = logging.getLogger("image_url_resolver")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler("url_resolution.log", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False

    logger.info("Attempting to resolve URL: %s", image_url)

    try:
        parsed = parse_image_url(image_url)
        validate_parsed_url(parsed)

        resolved = image_url.strip()
        logger.info("URL resolved successfully: %s", resolved)

        return resolved

    except Exception:
        logger.exception("Failed to resolve URL: %s", image_url)
        raise
